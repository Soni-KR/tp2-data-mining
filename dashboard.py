import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from fraud_core import ARTIFACTS, CONFIG, FEATURES, decisions, metrics, records, validate_frame
from client import score_frame

API_URL = os.getenv('API_URL', 'http://localhost:8000')
st.set_page_config(page_title='Fraud Observatory | ENSI', layout='wide')
st.markdown('''<style>
[data-testid="stMainBlockContainer"] {padding: 3rem 2rem; max-width: 1600px;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
[data-testid="stMetricLabel"] p {white-space: normal;}
[data-testid="stHeading"] h2 {font-size: 1.4rem;}
@media (max-width: 1100px) {
  [data-testid="stHorizontalBlock"] {flex-wrap: wrap;}
  [data-testid="stColumn"] {flex: 1 1 260px; min-width: min(100%, 260px);}
  [data-testid="stColumn"]:has([data-testid="stMetric"]) {flex: 1 1 180px; min-width: min(100%, 180px);}
  [data-testid="stForm"] [data-testid="stColumn"] {flex: 1 1 180px; min-width: min(100%, 180px);}
}
@media (max-width: 600px) {
  [data-testid="stMainBlockContainer"] {padding: 2rem 1rem;}
}
</style>''', unsafe_allow_html=True)

@st.cache_data
def load_csv(name):
    return pd.read_csv(ARTIFACTS / name)

@st.cache_data
def comparison(data, k):
    rows, curves = [], []
    for name, col in [('Logistic Regression','lr_score'), ('XGBoost','xgb_score'), ('Isolation Forest','iso_score'), ('LOF','lof_score')]:
        scores = data[col].to_numpy()
        m = metrics(data, decisions(scores, 'budget', budget=k))
        rows.append(dict(Model=name, AP=average_precision_score(data.isFraud, scores), ROC_AUC=roc_auc_score(data.isFraud, scores), Precision_at_k=m['precision'], Recall_at_k=m['recall'], F2_at_k=m['f2'], MCC_at_k=m['mcc']))
        p, r, _ = precision_recall_curve(data.isFraud, scores)
        # Retain genuine curve coordinates; thin only for rendering.
        idx = np.unique(np.r_[np.arange(0, len(p), max(1, len(p)//1500)), len(p)-1])
        curves.append(pd.DataFrame({'Recall':r[idx], 'Precision':p[idx], 'Model':name}))
    return pd.DataFrame(rows), pd.concat(curves)


def chart(fig):
    fig.update_layout(template='plotly_dark', paper_bgcolor='#151F32', plot_bgcolor='#151F32', margin=dict(l=15,r=15,t=40,b=15), font=dict(size=14, color='#E2E8F0'), legend_title_text='', colorway=['#38BDF8','#FBBF24','#A78BFA','#2DD4BF'])
    st.plotly_chart(fig, width='stretch', theme=None)


def metric_row(values):
    for col, (name, value) in zip(st.columns(len(values)), values):
        col.metric(name, value)


def notes(*items):
    with st.expander('About this analysis'):
        for item in items:
            st.caption(item)


def confusion(m):
    if 'tp' in m:
        chart(px.imshow([[m['tn'],m['fp']],[m['fn'],m['tp']]], x=['PASS','REVIEW'], y=['Legitimate','Fraud'], text_auto=True, color_continuous_scale='Blues', labels=dict(x='Decision',y='Ground truth'), title='Confusion matrix'))


def local_explanation(frame, key):
    if frame.empty:
        st.info('Load transactions to inspect local explanations.')
        return
    positions = list(np.argsort(-frame.xgb_score.to_numpy(), kind='stable')) if 'xgb_score' in frame else list(range(len(frame)))
    pos = st.selectbox('Transaction to explain', positions, format_func=lambda i: f"Row {i+1} | ID {frame.iloc[i].get('TransactionID', 'unavailable')}", key=key)
    if st.button('Compute local TreeSHAP', key=key+'_button'):
        try:
            with st.spinner('Computing explanation for one transaction…'):
                response = requests.post(API_URL+'/explain', json={'features':records(frame.iloc[[pos]][FEATURES])[0]}, timeout=60)
                response.raise_for_status()
                st.session_state[key+'_result'] = (pos, response.json())
        except requests.RequestException as exc:
            st.error('Explanation request failed. Check the API connection. '+str(exc))
    cached = st.session_state.get(key+'_result')
    if not cached or cached[0] != pos:
        return
    result = cached[1]
    metric_row([('Risk score',f"{result['risk_score']:.4f}"), ('Decision',result['decision'])])
    contributions = sorted(result['contributions'], key=lambda x:abs(x['shap_value']), reverse=True)
    top = contributions[:10]
    other = sum(x['shap_value'] for x in contributions[10:])
    labels = ['Baseline']+[x['feature'] for x in top]+['Other features','Model margin']
    vals = [result['baseline_margin']]+[x['shap_value'] for x in top]+[other,0]
    chart(go.Figure(go.Waterfall(x=labels, y=vals, measure=['absolute']+['relative']*(len(top)+1)+['total'], connector={'line':{'color':'#64748b'}})).update_layout(title='Additive local explanation · log-odds', xaxis_tickangle=-35))
    reconstructed = result['baseline_margin'] + sum(x['shap_value'] for x in contributions)
    if not np.isclose(reconstructed,result['raw_margin'],atol=2e-5) or not np.isclose(result['reconstructed_score'],result['risk_score'],atol=2e-6):
        st.error('SHAP reconstruction check failed.')
    with st.expander('Reason codes'):
        st.dataframe(pd.DataFrame(result['reason_codes']), hide_index=True, width='stretch')
    notes('SHAP in log-odds: positive raises risk; negative lowers it. Anonymized features have no verified business meaning. Explanations are not causal.',
          f'Baseline + contributions = {reconstructed:.4f}; sigmoid(margin) = {result["reconstructed_score"]:.4f}.')

st.sidebar.title('Fraud Observatory')
st.sidebar.caption('ENSI · Data Mining Challenge 2')
page = st.sidebar.radio('Workspace', ['Overview & EDA','Transaction Monitoring','Decision Lab','Model Evaluation','Explainability'])
try:
    health = requests.get(API_URL+'/health',timeout=3)
    health.raise_for_status()
    st.sidebar.caption('● API connected')
except requests.RequestException:
    st.sidebar.warning('API offline · saved analysis available')
st.title(page)
try:
    validation = load_csv('dashboard_validation.csv')
except (OSError, ValueError) as exc:
    st.error('Cannot load validation artifacts: '+str(exc))
    st.stop()

if page == 'Overview & EDA':
    st.caption('Validation set')
    metric_row([('Validation transactions',f'{len(validation):,}'),('Fraud',f'{validation.isFraud.sum():,}'),('Validation fraud rate',f'{validation.isFraud.mean():.2%}'),('Input features',str(len(FEATURES)))])
    distributions,timeline,categories=st.tabs(['Distributions','Time','Categories'])
    with distributions:
        a,b=st.columns(2)
        with a:
            counts=validation.isFraud.map({0:'Legitimate',1:'Fraud'}).value_counts().rename_axis('Class').reset_index(name='Transactions')
            chart(px.bar(counts,x='Class',y='Transactions',text_auto=True,title='Class imbalance'))
        with b:
            cap=st.slider('Amount display percentile',90,100,99)
            amounts=validation[validation.TransactionAmt <= validation.TransactionAmt.quantile(cap/100)].copy()
            amounts['Class']=amounts.isFraud.map({0:'Legitimate',1:'Fraud'})
            chart(px.histogram(amounts,x='TransactionAmt',color='Class',nbins=60,barmode='overlay',title=f'Transaction amounts · lower {cap}%'))
    with timeline:
        periods=validation.assign(relative_day=(validation.TransactionDT//86400).astype(int)).groupby('relative_day').isFraud.agg(['mean','size']).reset_index()
        chart(px.line(periods,x='relative_day',y='mean',markers=True,hover_data=['size'],title='Fraud rate by elapsed day',labels={'mean':'Fraud rate','relative_day':'Elapsed day from unknown dataset origin'}))
    with categories:
        a,b=st.columns(2)
        for col,field in [(a,'ProductCD'),(b,'card4')]:
            with col:
                grouped=validation.assign(**{field:validation[field].fillna('(missing)')}).groupby(field).isFraud.agg(['mean','size']).reset_index()
                chart(px.bar(grouped,x=field,y='mean',hover_data=['size'],title=f'Fraud rate by {field}',labels={'mean':'Fraud rate'}))
    notes('Full dataset: 590,540 rows; approximately 3.50% fraud. Charts show validation only.',
          'Elapsed time has no verified calendar dates. Anonymous groups are not verified customers. Class imbalance makes accuracy misleading.')

elif page == 'Transaction Monitoring':
    source=st.radio('Input source',['Built-in demo','Upload CSV','Try one transaction'],horizontal=True)
    if source=='Upload CSV':
        with st.expander('CSV format & template'):
            st.dataframe(pd.DataFrame({'feature':FEATURES,'type':['numeric' if f in CONFIG['numerical'] else 'category' for f in FEATURES]}),hide_index=True)
            st.caption('All 68 columns required. Empty cells are imputed. Historical features must be precomputed from earlier events. Optional: TransactionID, binary isFraud and metadata.')
            st.download_button('Download template', (ARTIFACTS/'demo_transactions.csv').read_bytes(), 'demo_transactions.csv','text/csv')
    upload=st.file_uploader('Feature-ready transactions',type=['csv']) if source=='Upload CSV' else None
    frame=None
    submitted=False
    try:
        if source=='Built-in demo':
            frame=load_csv('demo_transactions.csv')
        elif source=='Try one transaction':
            samples=load_csv('reference.csv')
            with st.expander('Choose sample'):
                sample_number=int(st.number_input('Sample row',min_value=1,max_value=len(samples),value=1))
            sample=samples.iloc[sample_number-1]
            entered={}
            common=['TransactionAmt','ProductCD','card4','card6']
            history=['time_sin','time_cos','time_since_prev','tx_1h','tx_24h','amt_ratio_30d']

            def fields(names):
                columns=st.columns(2)
                for i,name in enumerate(names):
                    value=sample[name]
                    entered[name]=columns[i%2].text_input(name,value='' if pd.isna(value) else str(value),key=f'sample_{sample_number}_{name}')

            with st.form('single_transaction'):
                st.header('Transaction details')
                fields(common)
                with st.expander('Historical features'):
                    st.caption('Prefilled history · not recalculated when editing.')
                    fields(history)
                with st.expander('Other model features'):
                    fields([f for f in FEATURES if f not in common+history])
                submitted=st.form_submit_button('Score sample',type='primary')
            if submitted:
                st.session_state.pop('sample_prediction',None)
                frame=validate_frame(pd.DataFrame([{f:(None if not entered[f].strip() else entered[f].strip()) for f in FEATURES}]))
                frame['input_source']=f'Edited reference sample {sample_number}'
        elif upload:
            # Reject duplicate headers before pandas can silently mangle them.
            import csv, io
            names=next(csv.reader(io.StringIO(upload.getvalue().decode('utf-8-sig'))),[])
            if len(names)!=len(set(names)):
                raise ValueError('Duplicate column names are not allowed.')
            frame=pd.read_csv(upload)
        if frame is not None:
            validate_frame(frame)
            if 'TransactionID' in frame and frame.TransactionID.duplicated().any():
                st.warning('Duplicate transaction IDs retained. Use row positions to distinguish records.')
            if source!='Try one transaction':
                st.caption(f'{len(frame):,} transactions')
                with st.expander('Preview data'):
                    st.dataframe(frame.head(10),hide_index=True)
            if submitted if source=='Try one transaction' else st.button('Score transactions',type='primary'):
                with st.spinner('Scoring…'):
                    scored=score_frame(frame,API_URL)
                st.session_state['scored']=scored
                st.session_state['scored_source']=source
                for k in list(st.session_state):
                    if k.endswith('_result'):
                        st.session_state.pop(k,None)
                if source=='Try one transaction':
                    st.session_state['sample_prediction']=(sample_number,scored.iloc[0].to_dict())
    except (ValueError, OSError, UnicodeError, requests.RequestException) as exc:
        st.error(str(exc))
    scored=st.session_state.get('scored')
    if source=='Try one transaction':
        prediction=st.session_state.get('sample_prediction')
        if prediction and prediction[0]==sample_number:
            st.header('Result')
            result=prediction[1]
            a,b=st.columns(2)
            a.metric('Risk score',f"{result['xgb_score']:.4f}",help='Model risk score; not a calibrated probability.')
            b.metric('Decision','REVIEW' if result['flagged'] else 'PASS',help='Deployed threshold: 0.39.')
    elif scored is not None and st.session_state.get('scored_source')==source:
        st.header('Results')
        metric_row([('Transactions',f'{len(scored):,}'),('Deployed alerts',f'{scored.flagged.sum():,}'),('Alert rate',f'{scored.flagged.mean():.2%}')])
        with st.expander('Filter & sort'):
            minimum=st.slider('Minimum displayed risk score',0.,1.,0.,.01)
            alerts_only=st.checkbox('Show alerts only')
            ordering=st.selectbox('Sort by',['Risk descending','Risk ascending','Original row order'])
        filtered=scored[(scored.xgb_score>=minimum)&(scored.flagged if alerts_only else True)]
        filtered=filtered.sort_values('row_position' if ordering=='Original row order' else 'xgb_score',ascending=ordering!='Risk descending',kind='stable')
        display_columns=[c for c in ['row_position','TransactionID','xgb_score','flagged','TransactionAmt','ProductCD','card4','isFraud'] if c in filtered]
        st.dataframe(filtered[display_columns],hide_index=True,width='stretch')
        with st.expander('Complete scored records and input features'):
            st.dataframe(filtered,hide_index=True,width='stretch')
        st.download_button('Download complete scored CSV',scored.to_csv(index=False),'scored_transactions.csv','text/csv')
        with st.expander('Explain a transaction'):
            local_explanation(scored,'monitor')

elif page == 'Decision Lab':
    source=st.selectbox('Evaluation population',['Saved validation']+(['Last scored batch'] if 'scored' in st.session_state else []))
    data=validation if source=='Saved validation' else st.session_state['scored']
    mode=st.radio('Decision policy',['Threshold','Analyst budget'],horizontal=True)
    a,b,c=st.columns(3)
    threshold=a.slider('Scenario risk threshold',0.,1.,.39,.01,disabled=mode!='Threshold')
    budget=int(b.number_input('Maximum investigations',min_value=0,value=10000,step=100,disabled=mode!='Analyst budget'))
    cost=c.number_input('Investigation cost (illustrative units)',min_value=0.,value=5.,step=1.)
    flagged=decisions(data.xgb_score,'threshold' if mode=='Threshold' else 'budget',threshold,budget)
    m=metrics(data,flagged,cost)
    metric_row([('Transactions',f'{len(data):,}'),('Alerts',f"{m['alerts']:,}"),('Alert rate',f"{m['alert_rate']:.2%}"),('Investigation cost',f"{m['investigation_cost']:,.2f}")])
    performance,costs,curves,policy_table=st.tabs(['Performance','Costs','Curves','Policies'])
    with performance:
        if 'precision' in m:
            metric_row([(k.upper(),f'{m[k]:.4f}') for k in ['precision','recall','f2','mcc']])
            confusion(m)
        else:
            st.warning('Unlabeled data · alert counts and investigation cost only.')
    with costs:
        if 'total_cost' in m:
            metric_row([('Missed-fraud loss',f"{m['missed_fraud_loss']:,.2f}"),('Total estimated cost',f"{m['total_cost']:,.2f}"),('No-model baseline',f"{m['no_model_cost']:,.2f}"),('Cost reduction',f"{m['cost_reduction']:.2%}" if m['cost_reduction'] is not None else 'N/A')])
        else:
            st.caption('Total business cost also requires complete finite, nonnegative amounts.')
    with curves:
        threshold_rows=[dict(threshold=t,**metrics(data,decisions(data.xgb_score,threshold=t),cost)) for t in np.linspace(0,1,51)]
        budgets=np.unique(np.r_[0,np.linspace(0,len(data),51,dtype=int),min(budget,len(data)),min(10000,len(data))])
        budget_rows=[dict(budget=int(k),**metrics(data,decisions(data.xgb_score,'budget',budget=k),cost)) for k in budgets]
        td,bd=pd.DataFrame(threshold_rows),pd.DataFrame(budget_rows)
        cost_key='total_cost' if 'total_cost' in td else 'investigation_cost'
        a,b=st.columns(2)
        with a:
            chart(px.line(td,x='threshold',y=cost_key,title='Cost versus threshold'))
        with b:
            chart(px.line(bd,x='budget',y=cost_key,title='Cost versus analyst budget'))
        if 'precision' in td:
            chart(px.line(td,x='threshold',y=['precision','recall'],title='Precision / recall versus threshold'))
    with policy_table:
        policies=[]
        for name,pred in [('Reference threshold 0.39',decisions(data.xgb_score)),('Reference budget 10,000',decisions(data.xgb_score,'budget',budget=10000)),('Current scenario',flagged)]:
            policies.append(dict(policy=name,**metrics(data,pred,cost)))
        st.header('Policy comparison')
        st.dataframe(pd.DataFrame(policies),hide_index=True,width='stretch')
    notes('Scenarios only: deployed threshold stays 0.39. Uploads are exploratory.',
          'Illustrative cost = missed fraud amounts + investigation cost × alerts. Budget ties use row order; k is capped at row count.')

elif page == 'Model Evaluation':
    common=validation[validation.lof_score.notna()].copy()
    st.caption(f'Shared validation set · {len(common):,} rows')
    model_tab,validation_tab,test_tab=st.tabs(['Model comparison','Full validation','Final test'])
    with model_tab:
        k=int(st.number_input('Shared analyst budget k',min_value=0,max_value=len(common),value=min(100,len(common))))
        table,curves=comparison(common,k)
        st.dataframe(table,hide_index=True,width='stretch')
        chart(px.line(curves,x='Recall',y='Precision',color='Model',title='Precision–recall curves · common validation cohort').add_hline(y=common.isFraud.mean(),line_dash='dash',annotation_text='Fraud prevalence baseline'))
        chosen=st.selectbox('Confusion matrix model',table.Model)
        score_col={'Logistic Regression':'lr_score','XGBoost':'xgb_score','Isolation Forest':'iso_score','LOF':'lof_score'}[chosen]
        confusion(metrics(common,decisions(common[score_col],'budget',budget=k)))
    with validation_tab:
        st.header('XGBoost · full validation')
        metric_row([('Validation AP',f'{average_precision_score(validation.isFraud,validation.xgb_score):.4f}'),('Validation ROC-AUC',f'{roc_auc_score(validation.isFraud,validation.xgb_score):.4f}')])
    with test_tab:
        st.header('Final test')
        summary=json.loads((ARTIFACTS/'test_summary.json').read_text())
        metric_row([('Test transactions',f"{summary['n_transactions']:,}"),('Test AP',f"{summary['pr_auc']:.4f}"),('Test ROC-AUC',f"{summary['roc_auc']:.4f}"),('No-model cost',f"{summary['no_model_cost']:,.2f}")])
        st.dataframe(pd.DataFrame(summary['policies']).T, width='stretch')
    notes('AP = Average Precision (PR-AUC). Higher scores mean greater risk. Model comparisons use the same rows and top-k policy with stable ties. Precision captures false-alert burden.',
          'Validation → test: AP 0.538 → 0.438; ROC-AUC 0.913 → 0.886. Reduced temporal generalization; cause unverified. Policies selected on validation only.',
          'Aggregate test results only; no test curves or matrices. Test investigation cost: 5 units.')

else:
    global_tab,local_tab=st.tabs(['Global importance','Transaction'])
    with global_tab:
        importance=load_csv('global_shap_importance.csv').sort_values('mean_abs_shap',ascending=False)
        n=st.slider('Global features shown',5,40,20,help='Mean absolute SHAP in log-odds. Transformed features; magnitude, not direction or causality.')
        chart(px.bar(importance.head(n).iloc[::-1],x='mean_abs_shap',y='feature',orientation='h',title='Global feature importance · mean |SHAP|',labels={'mean_abs_shap':'Mean absolute log-odds contribution'}))
    with local_tab:
        st.header('Transaction explanation')
        sources=['Feature-ready demo','Reference transactions']+(['Last scored batch'] if 'scored' in st.session_state else [])
        selected=st.selectbox('Explanation source',sources)
        data=load_csv('demo_transactions.csv') if selected==sources[0] else load_csv('reference.csv') if selected=='Reference transactions' else st.session_state['scored']
        local_explanation(data,'explain_'+selected)

