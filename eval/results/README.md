# 真实模型验证记录

[English](README.en.md) · 中文

## 阶段6：三组长对话

运行器：[run_long_conversations.py](../run_long_conversations.py)。使用项目配置的真实DeepSeek API、联网搜索及现有中文知识索引；每组独立初始化模拟业务库，复制知识目录数据库，共享只读E5模型和Qdrant索引，不操作日常业务库。

```bash
.venv/bin/python eval/run_long_conversations.py --report eval/results/long-conversations-new.json
```

该命令会产生模型和搜索调用费用。可用`--group 1`选择单组；任一组断言失败时退出码为1。原请求重放、选单、付款方式选择和动作确认均走FastAPI接口，完整保存回复、计划、轨迹、交互、耗时及断言。

| 组别 | 轮数 | 覆盖 |
|---|---:|---|
| 1 | 21 | 系统帮助；三次观察、明确设置、临时要求、购物偏好修改/撤销/重新启用；订单协作；商品与政策RAG、概览和缺失资料 |
| 2 | 17 | 订单列表/详情/状态/物流；支付与退款查询、选单、取消、逐笔确认、重放、重启恢复、依赖条件及独立任务部分失败 |
| 3 | 16 | 英文提问查询中文资料并英文解释；客户隔离；支付；搜索与证据边界；投诉咨询、正式工单与状态追问；不支持操作 |

最终报告见[long-conversations-final.json](long-conversations-final.json)：三组54轮、11次选择/确认提交通过，覆盖全部13项登记能力；离线回归176项通过。汇总采用各组完整通过的运行并保留来源文件名，不是首次无失败运行：[中文组](long-conversations-final-1b.json)、[支付退款组](long-conversations-final-12.json)、[英文组](long-conversations-final-3b.json)。首测及中间复测均保留：[initial](long-conversations-initial.json)、[retest](long-conversations-retest.json)、[final-13](long-conversations-final-13.json)、[final-3](long-conversations-final-3.json)。自动断言通过后，人工审阅仍发现空资料综合、否定句误拦及子问题重复，另作修复；最终计划均通过最新结构校验。

补充的真实Service＋Streamlit AppTest交互见[supplement](long-conversations-supplement.json)：支付方式默认未选、选择、英文切换、取消、资料筛选和清除；其中三条临时“详细一点”未提取长期偏好，不能据此声称验证三次观察。最新子问题聚焦改动使用英文组已保存的实际任务资料，另外调用两次真实API验证，见[synthesis-followup](long-conversations-synthesis-followup.json)；原始长对话回复未改写。

修复包括：格式修复消息不再充当用户输入；明确无编号退款走选单；加强回复语言和平台投诉资料路由；引用格式失败有限重试；工单状态咨询拦截重复建单；不推断币种或平台限制；综合说明必须有资料依赖、聚焦各自子问题并避免重复展示上游搜索回答；工单声明校验保留正常否定句。观察偏好与明确长期设置分开，严格保留当前原话校验；无效候选仍单独丢弃，不阻断业务。

覆盖以已登记能力、业务状态和交互路径为准。金额操作仅改变本地模拟记录；工单不通知真人。自然语言回答经抽查，程序断言不证明每句忠实或任意表达都可靠；固定业务回执仍包含中文。该轮未验证生产资金、部署、多实例并发或全部设备视觉效果。

## 阶段4历史记录

全部使用独立临时合成库和配置的DeepSeek模型，不读取现有业务库。每个场景、每次重复均重新初始化合成样例。

| 报告 | 结果 | 说明 |
|---|---|---|
| [initial.json](initial.json) | 14/16 | 首次基础场景；10政策误分类、15任务成功值错误 |
| [boundary-initial.json](boundary-initial.json) | 35/36 | 首次重复组合；20未付款条件错误使用unpaid |
| [baseline-final.json](baseline-final.json) | 22/22 | 修复后16组基础及6组泛化组合 |
| [boundary-final.json](boundary-final.json) | 36/36 | 11～22号场景各3次独立运行 |

修复采用通用政策路由约束、任务状态与业务状态枚举校验、跨任务资料解释。未添加案例文本匹配的业务分支。

最终离线回归141项通过；基础及组合报告22/22，重复报告36/36。每次记录含完整计划、工具轨迹、选择、动作、耗时和错误；初始失败未覆盖或删除。在线测试按操作覆盖、只读无写、归属、支付后资格及状态进行程序断言，不能由有限样本推导任意输入正确率。
