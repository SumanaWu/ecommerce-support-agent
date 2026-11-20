"""Presentation-only translations; stored business data and messages stay unchanged."""
import streamlit as st

EN = dict(line.split('|', 1) for line in '''电商订单与售后助手|Ecommerce Support Assistant
订单与售后 · 多专家协作 · 客户记忆|Orders and after-sales · Agent collaboration · Customer memory
本地模拟环境：支付与退款仅修改样例数据。|Local simulation: payments and refunds only update sample data.
客户编号|Customer ID
会话编号|Session ID
新建会话|New conversation
会话详情|Conversation details
最近讨论订单|Recent orders
展示轮次|Displayed turns
累计模型 Token|Total model tokens
当前为客户独立对话页面；仅可访问本人订单。客户编号用于本地演示身份切换，并非登录认证。|Only this customer's orders are accessible. Editable IDs simulate identities; they are not authentication.
客服对话|Support chat
记忆与客户偏好|Memory and preferences
平台资料详情|Knowledge library
使用说明与多轮测试|Guide and acceptance scenarios
你好，需要什么帮助？|Hello, how can I help?
可以查询订单、了解平台政策，或告诉我你的长期购物偏好。|Ask about an order, explore policies, or share a lasting shopping preference.
例如：查询订单 1001 的详情|For example: show details of order 1001
继续处理|Continue processing
查看我的全部订单|Show all my orders
准备所选订单退款|Prepare selected refunds
取消选单|Cancel selection
确认付款|Confirm payment
确认办理|Confirm refund
取消本项|Cancel this action
取消支付|Cancel payment
支付方式|Payment method
请选择支付方式|Choose a payment method
选择方式仅准备方案；点击确认付款后才执行。|Choosing a method prepares the action. Only confirmation executes payment.
请先完成其他必要选择，再按顺序确认。|Complete the required selections, then confirm each action in order.
等待前一项操作确认或取消。|Waiting for the previous action to be confirmed or cancelled.
请先完成确认或取消|Please confirm or cancel the pending action
正在处理|Processing
任务进度|Task progress
工具执行轨迹|Tool trace
办理明细|Action details
协作处理：|Agents involved: 
主对话层|Conversation layer
处理专家：|Agent: 
已保留原有购物偏好；如需修改，请明确告诉我。|Your saved shopping preference was kept. Explicitly request a change to update it.
状态：等待前一项处理，尚未核实|Status: waiting for the previous action; not yet verified
状态：等待前置任务确定订单和处理条件|Status: waiting for prerequisite orders and conditions
状态：已取消，未支付|Status: cancelled; no payment made
待确认，尚未执行|Awaiting confirmation; not executed
查询订单列表|List orders
查询订单详情|Order details
查询订单状态|Order status
查询物流|Shipment status
查询支付|Payment status
准备支付|Prepare payment
准备退款|Prepare refund
检查退款资格|Refund eligibility
检索平台资料|Search knowledge
综合资料|Synthesize evidence
创建本地工单|Create local ticket
等待执行|Pending
已选择|Selected
已完成|Completed
处理失败|Failed
未办理|Rejected
已跳过|Skipped
已取消|Cancelled
等待选择|Awaiting selection
等待确认|Awaiting confirmation
订单专家|Order agent
支付专家|Payment agent
退款专家|Refund agent
知识检索|Knowledge search
系统帮助|System help
联网搜索|Web search
本地工单|Local ticket
退款申请|Refund request
支付|Payment
任务|Task
操作|Operation
订单|Order
状态|Status
依赖|Dependencies
条件|Conditions
短期记忆 → 长期记忆|From conversations to lasting preferences
从已完成的用户对话提取回复与购物偏好；同一偏好的证据按不同请求计数，刷新或重试不增加次数。|Preferences are extracted from completed turns. Refreshing or retrying does not add evidence.
偏好分类|Preference category
全部|All
回复偏好|Response preferences
购物偏好|Shopping preferences
记忆状态|Memory status
当前记忆|Current memories
已撤销|Revoked
待转换|Candidates
长期记忆|Active memories
待转换的习惯|Preferences under observation
已保存的长期记忆|Saved preferences
已撤销的记忆|Revoked preferences
暂无待转换候选。普通聊天、仅本次要求和手动保存表单不会自动累加为习惯证据。|No candidates. Casual chat, one-off requests and manual forms do not automatically add observations.
暂无自动提取的长期记忆。|No automatically saved preferences.
暂无撤销记录。|No revoked preferences.
当前仍是候选，不作为已生效的长期偏好。|Still a candidate; not used as an active preference.
已停止使用；旧证据不会自动恢复这条记忆。|No longer used. Old evidence cannot automatically reactivate it.
用户明确要求，已直接保存，无需累计三次。|Saved from an explicit request; three observations are not required.
忘记此偏好|Forget this preference
转换规则与可复制的表达|Rules and examples
观察路径：出现一次 → 待转换；出现两次 → 继续观察；累计三次 → 转为长期记忆。仅统计模型已提取并保存的同类、同值证据，不是把所有重复文字都计数。|One observation creates a candidate; two remain under observation; three activate it. Only extracted, saved evidence for the same category and value counts.
已有明确设置或已撤销的偏好，不会被观察记录自动覆盖；涉及业务确认的对话在本轮操作结束后才写入记忆。|Observations do not override explicit or revoked preferences. Memory is written after the entire action flow ends.
购物偏好生效后，需明确要求修改；临时购买需求不会自动覆盖。|Active shopping preferences require an explicit change. Temporary requests do not override them.
请记住，我买耳机通常预算在500元以内。|Remember: I usually budget up to 500 yuan for headphones.
直接保存长期偏好：|Save a lasting preference:
以后请先给结论再解释，请记住这个回复偏好。|Remember to give the conclusion before the explanation in future replies.
只对当前回答生效，不应形成永久偏好：|Apply only to this answer:
这一次请详细解释，不要把它保存为长期习惯。|Explain in detail this time; do not save it as a lasting preference.
撤销已经保存的偏好：|Revoke a saved preference:
请忘记关于回答详细程度的长期偏好。|Forget my preference about response detail.
最近对话的提取与转换记录|Recent extraction records
已检查（实际状态见上方）|Checked; see current state above
待写入，可重试原请求|Write pending; retry the original request
本轮未完成，尚未写入|Turn incomplete; not written yet
无可用提取记录|No extraction record
本轮没有长期记忆候选|No memory candidates in this turn
用户原话|User evidence
偏好类别|Category
提炼内容|Extracted preference
识别方式|Origin
明确要求|Explicit request
观察习惯|Observation
撤销记忆|Revoke memory
保存或累计证据|Save or count evidence
处理结果|Result
当前会话暂无对话。转换记录来自实际请求，不展示虚构的候选。|No conversation yet. Records reflect real requests; no sample candidates are inserted.
手动保存的客户偏好|Manual preferences
回答详略|Response detail
组织方式|Response structure
回复语言|Response language
语气风格|Tone
购物 · 品牌|Shopping · Brand
购物 · 品类|Shopping · Category
购物 · 预算|Shopping · Budget
联系偏好|Contact preference
保存客户偏好|Save preferences
偏好已保存，可跨会话读取。|Preferences saved for future conversations.
跨会话档案原始数据|Raw customer profile
简洁，先结论后解释|Concise, conclusion first
优先站内文字联系|Prefer in-platform text contact
平台资料|Knowledge library
虚构平台资料；规则解释不等于本地工具已具备对应办理能力。|Fictional reference material. Explaining a policy does not implement its business rules.
资料类型|Document type
平台政策|Platform policies
商品资料|Product documents
按标题、商品名称或编号查找资料|Search by title, product name or ID
分类筛选|Category filter
清除筛选|Clear filters
上一页|Previous
下一页|Next
查看文档|Open document
知识库为空，请停止应用后通过目录导入命令导入资料。|The library is empty. Stop the app and run the import command.
没有符合筛选条件的资料，请调整或清除筛选。|No matching documents. Adjust or clear the filters.
处理失败：|Processing failed: 
执行失败：|Execution failed: 
订单 |Order 
**订单 |**Order 
 · 金额 | · Amount 
 · 支付**| · Payment**
待支付金额：|Amount due: 
支付方式：|Payment method: 
状态：|Status: 
已选 |Selected: 
 笔| orders
协作任务 · |Tasks · 
任务 |Task 
来源原话（|Evidence (
 条）| records)
已出现 |Observed 
 次 / | / 
 次 · 还差 | times · 
 次转为长期记忆| more required for activation
已累计 |Accumulated 
 次证据，已转为长期记忆。| observations; preference activated.
观察到的习惯累计 |Activate observed preferences after 
 次才转为长期记忆；“以后都……”“请记住……”等明确要求可直接保存。| observations. Explicit lasting requests can be saved immediately.
第 |Observation 
 次 · | · 
共 |Total: 
 份 · 第 | documents · Page 
 页 · 索引状态：| · Index: 
分类：|Category: 
 · 版本：| · Version: '''.splitlines())

EN.update({'银行卡 / Card': 'Card', 'Visa卡': 'Visa', '支付宝': 'Alipay', '微信支付': 'WeChat Pay'})


def tr(text, language=None):
    if (language or st.session_state.get('ui_language', '中文')) != 'English' or not isinstance(text, str):
        return text
    if text in EN:
        return EN[text]
    # Translate template fragments only; user messages and evidence are never passed here.
    for source in sorted(EN, key=len, reverse=True):
        text = text.replace(source, EN[source])
    return text


def table_rows(rows):
    labels = {'偏好类别', '识别方式', '操作', '处理结果', '状态'}
    return [{tr(key): tr(value) if key in labels else value for key, value in row.items()} for row in rows]


def localize_format(function=str):
    language = st.session_state.get('ui_language', '中文')
    return lambda value: tr(function(value), language)
