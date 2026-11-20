"""Three long live-model conversations, isolated business data, read-only existing RAG index."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GROUPS = {
1: [
 ('你好，介绍一下如何使用这个客服系统，不查询订单、不办理业务。', []),
 ('请详细一点，解释新会话会保留哪些数据。这不是永久设置。', []),
 ('我平时偏好详细解释。', []),
 ('我平时偏好详细解释。', []),
 ('我平时偏好详细解释。', []),
 ('请记住，以后先结论后解释，并且始终用中文回答。', []),
 ('请记住，我平时偏好友好温和的语气。', []),
 ('请记住，我偏好品牌A，平时更关注耳机。', []),
 ('请记住，我购买耳机的长期预算是500元以内。', []),
 ('本次预算是300元，只是本次，不改变长期预算。', []),
 ('我最近有时想买800元的耳机，但现在不要修改长期偏好。', []),
 ('请把耳机长期预算明确改成800元以内。', []),
 ('请忘记我的品牌长期偏好。', []),
 ('请记住，以后品牌偏好改成品牌B。', []),
 ('查询1002的物流，再查询1003的付款状态。', ['order.shipment','payment.status']),
 ('请查1001完整详情，只查询。', ['order.details','payment.status']),
 ('请查Wireless Headphones的连接方式、续航和有线麦克风限制，引用实际资料。', ['knowledge.search','answer.explain']),
 ('请查平台无理由退货期限，以及定制商品的例外。', ['knowledge.search','answer.explain']),
 ('请介绍全部政策分类概览，并说明是否覆盖每条规则全部细节。', ['knowledge.search','answer.explain']),
 ('请查SKU-999材质和保修，找不到就明确说，不要替换成相似款。', ['knowledge.search','answer.explain']),
 ('请总结本会话已保存的回复和购物偏好，以及与业务权限的关系。', []),
],
2: [
 ('列出我自己的所有订单，再查询各笔付款状态。', ['order.list','payment.status']),
 ('只检查1001、1002、1003、1004、1005、1006能否退款，不申请。', ['refund.check']),
 ('我想申请退款，但先让我选具体订单，不要自动选最近订单。', ['refund.prepare']),
 ('查询1001与1006退款状态，说明申请已记录与实际到账的区别。', ['refund.check']),
 ('为1003准备支付，我自己选择方式，等我按钮确认。', ['payment.prepare']),
 ('重新查1003付款状态，核对取消后未发生付款。', ['payment.status']),
 ('为1003准备支付，付款成功后自动重查退款资格，只检查不申请。', ['payment.prepare','refund.check']),
 ('查询1003完整详情，解释placed和paid是否矛盾，以及实际支付方式。', ['order.details','payment.status']),
 ('请再为1003付一次钱。', ['payment.status']),
 ('仅当1002已签收才申请退款，若仍运输中只给物流。', ['order.shipment','refund.prepare']),
 ('查询不存在的9999详情，同时查询1007物流，一单失败不影响另一单。', ['order.details','order.shipment']),
 ('1003申请退款，原因是接口不适合我的设备，等我按钮确认。', ['refund.prepare']),
 ('再给1003重复申请退款，同时查1002物流。', ['refund.check','order.shipment']),
 ('请查1001、1003、1005退款状态，并区分已申请和已退款。', ['refund.check']),
 ('查询1007详情，再根据该商品说明书解释适用尺寸。', ['order.details','knowledge.search','answer.explain']),
 ('准备给1006退款，同时为已付款1001准备支付，每笔分别说明结果。', ['refund.prepare','payment.prepare']),
 ('只查询1002的订单状态，不查物流，不办理其他业务。', ['order.status']),
],
3: [
 ('Hello. Please explain how order queries differ from financial actions. Answer in English this turn.', []),
 ('Show my own orders and full details including payments for 1008, 1009 and 1010.', ['order.details','payment.status']),
 ('Show order 1001 details and its payment information, even though another customer owns it.', ['order.details','payment.status']),
 ('Does the Webcam 1080p in order 1008 support Windows Hello or 4K? Search the actual documentation first.', ['knowledge.search','answer.explain']),
 ('How many days after delivery can I request a no-reason return? Are customized products excluded?', ['knowledge.search','answer.explain']),
 ('Give me an overview of all platform policies.', ['knowledge.search','answer.explain']),
 ('Please prepare payment for 1008. I will choose the method and confirm myself.', ['payment.prepare']),
 ('Check 1008 payment details and refund eligibility, and track 1010. Do not refund.', ['payment.status','refund.check','order.shipment']),
 ('The service is disappointing, but only explain how to complain. Do not create a ticket or contact a human.', ['knowledge.search','answer.explain']),
 ('What is the weather today? Do not guess my city.', []),
 ('Beijing. Search current sources and distinguish forecast from observation.', ['search.web']),
 ('Search Beijing weather and track order1010. Does rain prove this order was delayed by weather?', ['search.web','order.shipment','answer.explain']),
 ('I now formally complain about the handling of order1010 and request a human. Please record my concern.', ['escalate.ticket']),
 ('Has that ticket notified a real human? Can you guarantee resolution today?', []),
 ('Cancel order1010, change its address and arrange pickup. Explain missing tools rather than pretending success.', []),
 ('Query my three orders again and summarize their current facts. Explain whether you can read customer_a preferences. No new ticket or financial action.', ['order.list']),
]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--group',type=int,nargs='+',default=[1,2,3])
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    live_data=Path(os.environ.get('SUPPORT_DATA_DIR', ROOT/'data')).resolve()
    assert (live_data/'knowledge.sqlite').exists(), 'Existing imported Chinese corpus required'
    folder=Path(tempfile.mkdtemp(prefix='support-long-live-'))
    os.environ['SUPPORT_DATA_DIR']=str(folder/'bootstrap')
    os.environ['SUPPORT_AGENT_DB_URL']='sqlite:///'+str(folder/'bootstrap'/'business.sqlite')
    from src import tools
    from src.backend.db import get_engine,get_session_factory,Ticket
    from src.backend.seed import seed
    from src.service import SupportService
    from src.knowledge import get_knowledge
    from src.runtime_context import active_customer
    import api
    from fastapi.testclient import TestClient
    api.service.close()
    import src.planning as planning
    original_model = planning.get_llm
    planning_outputs = []
    class RecordingModel:
        def invoke(self, messages):
            result = original_model().invoke(messages)
            planning_outputs.append(result.content)
            return result
    planning.get_llm = RecordingModel
    report={'groups':[], 'model':os.environ.get('DEEPSEEK_MODEL','deepseek-flash'),
            'corpus':'existing Chinese index; SQLite copied; vectors read-only', 'coverage':[]}
    path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
    def save():path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    for group in args.group:
        data=folder/f'group{group}';data.mkdir()
        shutil.copy2(live_data/'knowledge.sqlite',data/'knowledge.sqlite')
        (data/'models').symlink_to(live_data/'models',target_is_directory=True)
        # Share the read-only encoder, but keep the catalog and business state isolated.
        knowledge=get_knowledge(str(data));knowledge.model=get_knowledge(str(live_data)).encoder()
        engine=get_engine('sqlite:///'+str(data/'business.sqlite'));tools.configure_db(engine)
        with get_session_factory(engine)() as db:seed(db)
        svc=SupportService(data/'memory.sqlite');api.service=svc
        client=TestClient(api.app)
        user='customer_b' if group==3 else 'customer_a';session='long'
        record={'group':group,'turns':[],'checks':[]};report['groups'].append(record)
        def check(condition,label):
            record['checks'].append({'name':label,'passed':bool(condition)})
        def pay(oid):
            token=active_customer.set(user)
            try:return tools.get_payment_status(oid)
            finally:active_customer.reset(token)
        for number,(message,expected) in enumerate(GROUPS[group],1):
            rid=f'g{group}t{number}';turn={'number':number,'message':message,'events':[]};record['turns'].append(turn)
            started=time.monotonic()
            planning_outputs.clear()
            try:
                body={'user_id':user,'session_id':session,'request_id':rid,'message':message}
                http=client.post('/chat',json=body);turn['http']=http.status_code
                if http.status_code not in (200,202):
                    turn['graph_errors']=[str(t.error) for t in svc.graph.get_state(svc._config(user,session,rid)).tasks if t.error]
                assert http.status_code in (200,202),http.text
                r=http.json();turn['initial']=r
                actual={t['operation'] for t in r.get('tasks',[])}
                # A duplicate refund may be rejected during prepare or checked read-only.
                acceptable=set(expected)
                if group==2 and number==13 and 'refund.prepare' in actual:
                    acceptable.discard('refund.check')
                    acceptable.add('refund.prepare')
                check(acceptable<=actual,f'turn{number} expected operations {expected}')
                check(not any(t['status']=='failed' and not (t.get('facts') and all(f.get('error') for f in t['facts'])) for t in r.get('tasks',[])),f'turn{number} no technical task failure')
                replay=client.post('/chat',json=body)
                check(replay.json()==r,f'turn{number} original request replay')
                steps=0
                while r['status']!='completed':
                    assert r['status']!='processing','Processing incomplete'
                    steps+=1;assert steps<15,'Too many interaction steps'
                    operation='op'+uuid4().hex
                    event={'user_id':user,'session_id':session,'request_id':operation}
                    if r['pending_inputs']:
                        entry=r['pending_inputs'][0];event['input_id']=entry['input_id']
                        if entry['type']=='orders':
                            ids=[x['order_id'] for x in entry['candidates'] if x.get('eligible')][:2]
                            event.update(decision='submit',order_ids=ids)
                            endpoint=f'/requests/{rid}/select-orders'
                        else:
                            event.update(decision='submit',method='wechat_pay' if group==3 else 'paypal')
                            endpoint=f'/requests/{rid}/select-payment-method'
                            check(pay(entry['orders'][0]['order_id'])['status']=='pending','method choice does not pay')
                    else:
                        a=r['current_action'];assert a
                        cancel=(group==2 and number==5) or (group==2 and a['order_id']==1006)
                        event.update(decision='cancel' if cancel else 'confirm',action_id=a['action_id'])
                        endpoint=f'/requests/{rid}/resume'
                    if group==2 and number==7 and steps==2:
                        svc.close();svc=SupportService(data/'memory.sqlite');api.service=svc
                        check(svc.get_request(user,session,rid)['current_action']['action_id']==event['action_id'],'restart preserves pending action')
                    response=client.post(endpoint,json=event);assert response.status_code==200,response.text
                    r=response.json();turn['events'].append({'endpoint':endpoint,'input':event,'response':r})
                    check(client.post(endpoint,json=event).json()==r,'decision replay')
                turn['final']=r
                if group==1:
                    turn['memories']=svc.store.memories(user)
                    if number==5:
                        detail=next((m for m in turn['memories'] if m['key']=='detail'),{})
                        check(detail.get('status')=='active' and detail.get('origin')=='observed' and len(detail.get('evidence',[]))==3,'three observations activate memory')
                    if number==21:
                        check(all(value in r['reply'] for value in ['800','B','耳机']),'saved shopping preferences appear in summary')
                if group==2 and number==6:check(pay(1003)['status']=='pending','cancel did not pay')
                if group==2 and number==13:
                    token=active_customer.set(user)
                    try:check(tools.check_refund_eligibility(1003).get('reason_code')=='REFUND_IN_PROGRESS','duplicate refund did not create another application')
                    finally:active_customer.reset(token)
                if group==2 and number==8:check(pay(1003)['status']=='paid','payment persisted')
                if group==3 and number in [4,5,6]:
                    check(bool(r['sources']),'English knowledge answer has sources')
                if group==3 and number==3:
                    check(not any(f.get('amount') for t in r['tasks'] for f in t.get('facts',[])),'foreign order facts hidden')
                if group==3 and number==11:check(bool(r['sources']),'live search sources')
                if group==3 and number==15:
                    check(not any(symbol in r['reply'] for symbol in ['€','$','£']),'no invented currency')
                with tools._session() as db:
                    count=db.query(Ticket).count()
                    if group==3:check(count==(1 if number>=13 else 0),f'turn{number} explicit-only ticket')
            except Exception as exc:
                turn['error']=str(exc)
                print('ERROR',group,number,str(exc)[:300],flush=True)
                # Do not silently cancel failures; preserve exact graph state for diagnosis.
                save();break
            finally:
                turn['planner_outputs']=list(planning_outputs)
                turn['seconds']=round(time.monotonic()-started,2)
            save();print('group',group,'turn',number,'checks failed',sum(not c['passed'] for c in record['checks']),flush=True)
        if group==1 and len(record['turns'])==len(GROUPS[group]):
            items={m['key']:m for m in svc.store.memories(user)}
            check(all(key in items for key in ['format','language','tone','shopping_brand','shopping_category','shopping_budget']),'six explicit memory categories')
            check('800' in items.get('shopping_budget',{}).get('value',''),'explicit shopping budget update')
            check('B' in items.get('shopping_brand',{}).get('value',''),'revoked brand reactivated')
            svc.store.save_profile(user,'简洁','站内文字')
            check(svc.store.profile(user)['contact_preference']=='站内文字','manual profile save')
            check(client.get('/users/'+user+'/memories').status_code==200,'memory API read')
            check(client.delete('/users/'+user+'/memories/shopping_brand').status_code==200,'memory API revoke')
            rr=client.post('/chat',json={'user_id':user,'session_id':'new','request_id':'newsession','message':'请说明我已保存的长期耳机预算。'})
            record['new_session']=rr.json();check(rr.status_code==200,'new session with preferences')
            check(not svc.store.profile('customer_b'),'profile isolation')
            listing=client.get('/knowledge/documents',params={'kind':'product','limit':20}).json()
            check(listing['total']==200,'product catalog count')
            check(client.get('/knowledge/documents/'+listing['items'][0]['id']).status_code==200,'document full text API')
        record['operations']=sorted({t['operation'] for turn in record['turns'] for r in [turn.get('initial',{}),turn.get('final',{}),*[e['response'] for e in turn['events']]] for t in r.get('tasks',[])})
        report['coverage']=sorted(set(report['coverage'])|set(record['operations']))
        record['passed']=len(record['turns'])==len(GROUPS[group]) and not any(t.get('error') for t in record['turns']) and all(c['passed'] for c in record['checks'])
        svc.close();engine.dispose();save()
    print(json.dumps({'passed':[g['passed'] for g in report['groups']],'coverage':report['coverage'],'report':str(path)},ensure_ascii=False),flush=True)
    if not all(g['passed'] for g in report['groups']):
        raise SystemExit(1)


if __name__=='__main__':main()
