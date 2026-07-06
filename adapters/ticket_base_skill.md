## [退款纠纷] 商品质量与售后退款
Trigger: 文本包含退款、退钱、退货、赔偿、坏了、不能用、质量、破损、发霉、受潮、难吃、不新鲜、少送等售后词。
Action: core_intent 优先判为 退款纠纷。除明确订单号、金额、账号 ID 外，不要把口味、温度、时长、态度描述写入 entities。
Verification: entities 只保留订单号、6 位以上编号、金额；无这些信息时返回 []。

## [物流投诉] 配送延迟与服务态度
Trigger: 文本包含快递、物流、骑手、外卖、配送、送餐、送错、迟到、超时、晚了、慢、司机、态度、丢件。
Action: core_intent 优先判为 物流投诉，urgency_level 通常为 高。配送耗时如 20分钟、半小时、1个小时只是投诉证据，不是 entities。
Verification: 出现配送时长但没有订单号或金额时，entities 必须为 []。

## [账号封禁] 登录封禁与风控解封
Trigger: 文本包含账号、账户、封禁、封号、解封、登录、密码、冻结、异地、风控。
Action: core_intent 优先判为 账号封禁，urgency_level 为 高。只抽取明确账号 ID、编号或金额。
Verification: 把“马上解封、无法登录、异地登录”作为意图证据，不要写入 entities。

## [系统Bug] 系统故障与支付异常
Trigger: 文本包含系统、bug、崩溃、闪退、报错、打不开、卡住、页面、网络、验证码、支付失败。
Action: core_intent 优先判为 系统Bug，urgency_level 为 高。错误现象、页面名称、验证码失败不是 entities，除非文本中有明确编号或金额。
Verification: 对纯故障描述返回 entities=[]。

## [虚假宣传] 宣传不符与承诺落差
Trigger: 文本包含虚假、宣传、广告、图文不符、不符、假货、欺骗、夸大、承诺、活动、不值得信任。
Action: core_intent 优先判为 虚假宣传。entities 只保留明确金额、订单号或编号。
Verification: 不把“图文不符、广告夸大、承诺没兑现”写入 entities。

## [物流投诉] 实体边界：配送时长不是实体
Trigger: 文本包含晚了、迟到、超时、等了、送来、配送，并出现 20分钟、半小时、1个小时、几十分钟等时长。
Action: 时长只用于判断物流投诉和紧急程度，不作为 entities。没有订单号、金额、账号 ID 时 entities=[]。
Verification: “送来晚了20分钟”“等了半小时才到”应输出 entities=[]。
