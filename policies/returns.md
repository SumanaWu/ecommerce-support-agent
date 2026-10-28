---
policy_id: returns
version: local-v1
title: 本地样例退货与退款规则
effective_from: 2026-01-01
keywords: 退货,退款条件,退款规则,return,refund policy,eligibility
---
# 退货与退款资格
本地样例仅允许已送达（delivered）、且没有已有退款申请（退款状态为 none）的订单申请退款。
本阶段不校验退货期限、商品类别或包装状态，没有“所有订单都可退款”的规则。
对应的后端执行规则版本为 local-refund-v1。政策文档本身不能修改这段执行规则。

## 用户确认与资金边界
模型只能创建待确认提案。客户检查订单、CNY 样例金额和原因后，需独立点击“确认申请”。
确认仅把本地申请更新为 requested，不代表退款批准或真实资金到账。本地样例不承诺真实退款到账期限。
