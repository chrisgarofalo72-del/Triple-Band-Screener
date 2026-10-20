import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title='Triple EMA Band Stock Scanner', layout='wide')
st.title('Triple EMA Band Stock Scanner')
st.warning('Reconstruction / test strategy — NOT verified as the exact Facebook video formula. The video shows three EMAs and an EMA ribbon, but the input sources and precise trade rules are not confirmed.')
st.caption('Daily adjusted price data from Yahoo Finance via yfinance. Scanner uses completed historical daily bars; refresh after market close for final daily signals. Not financial advice.')
DEFAULT='AAPL MSFT NVDA GOOGL AMZN META TSLA AMD PLTR SOUN IONQ RGTI QBTS ORIC APD CRM AVGO INTC COIN NFLX MU SMCI AXON ARM'
with st.sidebar:
 st.header('Settings')
 tickers=st.text_area('Tickers (commas, spaces or newlines)',DEFAULT,height=170)
 length=st.number_input('EMA band length',min_value=2,max_value=200,value=20)
 lookback=st.number_input('Signals within last N sessions',min_value=1,max_value=30,value=3)
 signal_type=st.selectbox('Signal to scan',['Close crosses ABOVE upper EMA','Close crosses ABOVE center EMA','Close crosses ABOVE lower EMA','Close crosses BELOW lower EMA'])
 require_sma=st.checkbox('Require close above rising SMA 150',value=True)
 min_price=st.number_input('Minimum share price ($)',min_value=0.0,value=5.0)
 min_vol=st.number_input('Minimum 20-day average volume',min_value=0,value=500000,step=100000)
 latest_only=st.checkbox('Must still meet bullish condition',value=True)
 st.caption('Three band lines are EMA(high), EMA(close), EMA(low). This is an explicit assumption, not a claim about the video.')

symbols=list(dict.fromkeys(x.strip().upper() for x in tickers.replace(',',' ').split() if x.strip()))
@st.cache_data(ttl=900,show_spinner=False)
def load(ticker):
 d=yf.download(ticker,period='2y',interval='1d',auto_adjust=True,progress=False,threads=False)
 if d is None or d.empty:return None
 if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
 d=d[['Open','High','Low','Close','Volume']].dropna().copy()
 return d if len(d)>170 else None

def analyze(ticker):
 try:
  d=load(ticker)
  if d is None:return None,'Insufficient price history'
  for col,source in [('Top','High'),('Middle','Close'),('Bottom','Low')]:d[col]=d[source].ewm(span=length,adjust=False).mean()
  d['SMA150']=d['Close'].rolling(150).mean()
  d['AvgVol20']=d['Volume'].rolling(20).mean()
  target={'Close crosses ABOVE upper EMA':'Top','Close crosses ABOVE center EMA':'Middle','Close crosses ABOVE lower EMA':'Bottom','Close crosses BELOW lower EMA':'Bottom'}[signal_type]
  if 'BELOW' in signal_type: sig=(d.Close.shift(1)>=d[target].shift(1))&(d.Close<d[target])
  else:sig=(d.Close.shift(1)<=d[target].shift(1))&(d.Close>d[target])
  signal_dates=d.index[sig.fillna(False)]
  cutoff_dates=d.index[-lookback:]
  recent=[dt for dt in signal_dates if dt in cutoff_dates]
  if not recent:return None,None
  last=d.iloc[-1];prev=d.iloc[-6]
  if last.Close<min_price or last.AvgVol20<min_vol:return None,None
  if require_sma and not (last.Close>last.SMA150 and last.SMA150>prev.SMA150):return None,None
  if latest_only and ('BELOW' not in signal_type) and not last.Close>d.iloc[-1][target]:return None,None
  if latest_only and 'BELOW' in signal_type and not last.Close<d.iloc[-1][target]:return None,None
  hit=recent[-1]
  return {'Ticker':ticker,'Signal date':str(hit.date()),'Sessions ago':int(len(d)-1-d.index.get_loc(hit)),'Close':round(float(last.Close),2),'Band top':round(float(last.Top),2),'Band center':round(float(last.Middle),2),'Band bottom':round(float(last.Bottom),2),'SMA150':round(float(last.SMA150),2),'SMA150 rising':bool(last.SMA150>prev.SMA150),'Avg volume (20d)':int(last.AvgVol20),'Above top %':round(100*(last.Close/last.Top-1),2)},None
 except Exception as e:return None,f'{type(e).__name__}: {e}'

if st.button('Run scan',type='primary'):
 rows=[];errors=[]
 bar=st.progress(0,text='Scanning...')
 with ThreadPoolExecutor(max_workers=5) as pool:
  futures={pool.submit(analyze,s):s for s in symbols}
  for i,fut in enumerate(as_completed(futures),start=1):
   ticker=futures[fut]
   try:res,err=fut.result()
   except Exception as exc:res,err=None,str(exc)
   if res:rows.append(res)
   if err:errors.append((ticker,err))
   bar.progress(i/len(futures),text=f'{i} of {len(futures)} tickers checked')
 bar.empty()
 st.session_state['results']=pd.DataFrame(rows).sort_values(['Sessions ago','Ticker']) if rows else pd.DataFrame()
 st.session_state['errors']=errors

if 'results' in st.session_state:
 out=st.session_state['results']
 st.subheader(f'Matches: {len(out)} of {len(symbols)} scanned')
 if out.empty:st.info('No matches under these settings. Try removing the SMA filter or broadening your list; no matches does not mean no stocks in the market qualify.')
 else:
  st.dataframe(out,use_container_width=True,hide_index=True)
  st.download_button('Download matches CSV',out.to_csv(index=False).encode(),'triple_ema_band_matches.csv','text/csv')
  chosen=st.selectbox('Chart a matching stock',out.Ticker.tolist())
  d=load(chosen).tail(130).copy()
  for col,source in [('Top','High'),('Middle','Close'),('Bottom','Low')]:d[col]=d[source].ewm(span=length,adjust=False).mean()
  d['SMA150']=load(chosen)['Close'].rolling(150).mean().reindex(d.index)
  fig=go.Figure()
  fig.add_trace(go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close,name='Price'))
  for col,c in [('Top','#3b82f6'),('Middle','#2563eb'),('Bottom','#93c5fd'),('SMA150','#ef4444')]:
   fig.add_trace(go.Scatter(x=d.index,y=d[col],name=col,mode='lines',line=dict(color=c,width=1.6)))
  fig.update_layout(height=580,xaxis_rangeslider_visible=False,template='plotly_dark')
  st.plotly_chart(fig,use_container_width=True)
 if st.session_state.get('errors'):
  with st.expander(f'Data unavailable for {len(st.session_state["errors"])} tickers'):
   st.write(st.session_state['errors'])

st.divider()
st.markdown('**How this test signal works:** On a completed daily bar, yesterday’s close was at/below the selected EMA band, and today’s close finished above it (reverse for bearish breakdown). The scan checks the most recent N available sessions, not calendar days. **These rules are our reconstruction, not validated rules from the Facebook video.**')
