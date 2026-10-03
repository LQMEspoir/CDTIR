# File: src/tourism_ie/prompts.py
# Module responsibility: NER/NRE system prompt definition module. Returns system instructions that constrain the model output format according to the task and prompt profile.
# Main data flow: task + profile -> system prompt string.
# Reading guide: start with this file's public functions/classes, then follow the imports to common modules such as data_utils, modeling, and evaluation.
# Maintenance note: this version only adds explanatory comments; it does not modify existing expressions, control flow, default parameter values, or function call relationships.

# NER label set (translated): destination, departure, return, dining, lodging, product, transport, budget, duration, time, crowd, intensity, popularity, headcount, weather.
NER_LABELS = ["目的地", "出发地", "返程地", "餐饮", "住宿", "产品", "交通", "预算", "时长", "时间", "人群", "强度", "人气", "人数", "天气"]
# Relation label set (translated): visit order, travel time, length of stay, budget limit, per-capita budget, experience content.
RELATION_LABELS = ["游玩顺序", "出行时间", "停留时长", "预算限制", "人均预算", "体验内容"]

# You are an expert in text entity recognition.
# You need to extract entities of the following categories from the given sentence:
# - Location-related: destination, departure, return
# - Time-related: duration, time
# - Service-related: dining, lodging, product, transport
# - Attribute-related: budget, crowd, intensity, popularity, headcount, weather
# To extract the personalized intent needs in the user's travel planning.
# Please output in json format, e.g.
# {"entity_text": "Beijing", "entity_label": "destination"}
# Notes:
# 1. Every line of output must be a correct json string.
# 2. If no entity is found in the sentence, output "no entity found".
PAPER_NER_SYSTEM_PROMPT = """你是一个文本实体识别领域的专家。
你需要从给定句子中提取以下类别的实体：
- 地点相关：目的地、出发地、返程地
- 时间相关：时长、时间
- 服务相关：餐饮、住宿、产品、交通
- 属性相关：预算、人群、强度、人气、人数、天气
为了提取用户旅游规划中的个性化意图需求。
请以json格式输出，如
{\"entity_text\": \"北京\", \"entity_label\": \"目的地\"}
注意：
1.输出的每一行都必须是正确的json字符串。
2.如果句子中找不到任何实体，输出\"没有找到任何实体\"。"""

# Few-shot NER: adds one example for each of the 15 categories on top of paper_ner (the entities are brand-new examples that never appear in the dataset, to avoid leakage).
# Examples (translated):
# Text: I plan to go to Ejin to see the poplar forest
# Output: {"entity_text": "Ejin", "entity_label": "destination"}
# Text: Set off by train from Korla
# Output: {"entity_text": "Korla", "entity_label": "departure"}
# Text: After the itinerary, return to Hami
# Output: {"entity_text": "Hami", "entity_label": "return"}
# Text: Taste the local donkey-meat burger when you get there
# Output: {"entity_text": "donkey-meat burger", "entity_label": "dining"}
# Text: Want to stay a night in a treehouse
# Output: {"entity_text": "treehouse", "entity_label": "lodging"}
# Text: Want to try paragliding
# Output: {"entity_text": "paragliding", "entity_label": "product"}
# Text: Take the green train and slowly head over
# Output: {"entity_text": "green train", "entity_label": "transport"}
# Text: Budget is 8800 yuan
# Output: {"entity_text": "8800", "entity_label": "budget"}
# Text: Plan to play for half a month
# Output: {"entity_text": "half a month", "entity_label": "duration"}
# Text: Go around the winter solstice
# Output: {"entity_text": "winter solstice", "entity_label": "time"}
# Text: Is it suitable for honeymoon couples
# Output: {"entity_text": "honeymoon couple", "entity_label": "crowd"}
# Text: Want a leisurely tour pace
# Output: {"entity_text": "leisurely tour", "entity_label": "intensity"}
# Text: Want to find a niche, offbeat place
# Output: {"entity_text": "niche and offbeat", "entity_label": "popularity"}
# Text: Eight of us are going
# Output: {"entity_text": "eight people", "entity_label": "headcount"}
# Text: Want to pick a sunny day
# Output: {"entity_text": "sunny day", "entity_label": "weather"}
PAPER_NER_FEWSHOT_SYSTEM_PROMPT = PAPER_NER_SYSTEM_PROMPT + """

示例：
文本：我准备去额济纳看胡杨林
输出：{"entity_text": "额济纳", "entity_label": "目的地"}

文本：从库尔勒坐火车出发
输出：{"entity_text": "库尔勒", "entity_label": "出发地"}

文本：行程kill再返回哈密
输出：{"entity_text": "哈密", "entity_label": "返程地"}

文本：到了当地尝尝驴肉火烧
输出：{"entity_text": "驴肉火烧", "entity_label": "餐饮"}

文本：想住一晚树屋
输出：{"entity_text": "树屋", "entity_label": "住宿"}

文本：想体验一下滑翔伞
输出：{"entity_text": "滑翔伞", "entity_label": "产品"}

文本：坐绿皮火车慢慢晃过去
输出：{"entity_text": "绿皮火车", "entity_label": "交通"}

文本：预算准备8800元
输出：{"entity_text": "8800", "entity_label": "预算"}

文本：打算玩半个月
输出：{"entity_text": "半个月", "entity_label": "时长"}

文本：冬至那几天去
输出：{"entity_text": "冬至", "entity_label": "时间"}

文本：蜜月夫妻适合去吗
输出：{"entity_text": "蜜月夫妻", "entity_label": "人群"}

文本：想要休闲游的节奏
输出：{"entity_text": "休闲游", "entity_label": "强度"}

文本：想找个小众冷门的地方
输出：{"entity_text": "小众冷门", "entity_label": "人气"}

文本：我们八个人去
输出：{"entity_text": "八个人", "entity_label": "人数"}

文本：想挑个晴天去
输出：{"entity_text": "晴天", "entity_label": "天气"}"""

# Few-shot NER with one example per entity GROUP (Location/Time/Service/Attribute); each example covers ALL sub-classes of that group.
# Examples (translated):
# Text: Depart from Mohe to Baihaba, then return to Arctic Village after playing
# Output: {"entity_text": "Mohe", "entity_label": "departure"} / {"entity_text": "Baihaba", "entity_label": "destination"} / {"entity_text": "Arctic Village", "entity_label": "return"}
# Text: Go around the winter solstice and play for half a month
# Output: {"entity_text": "winter solstice", "entity_label": "time"} / {"entity_text": "half a month", "entity_label": "duration"}
# Text: Take a ferry to the resort house, taste the whole roast lamb, buy some local specialties
# Output: {"entity_text": "ferry", "entity_label": "transport"} / {"entity_text": "resort house", "entity_label": "lodging"} / {"entity_text": "whole roast lamb", "entity_label": "dining"} / {"entity_text": "local specialties", "entity_label": "product"}
# Text: Budget 8800, retired crowd leisurely tour, find a niche place, a party of three, pick a sunny day
# Output: {"entity_text": "8800", "entity_label": "budget"} / {"entity_text": "retired", "entity_label": "crowd"} / {"entity_text": "leisurely tour", "entity_label": "intensity"} / {"entity_text": "niche", "entity_label": "popularity"} / {"entity_text": "three people", "entity_label": "headcount"} / {"entity_text": "sunny day", "entity_label": "weather"}
PAPER_NER_FEWSHOT_4GROUP = PAPER_NER_SYSTEM_PROMPT + """

示例：
文本：从漠河出发去白哈巴，玩完返回北极村
输出：
{"entity_text": "漠河", "entity_label": "出发地"}
{"entity_text": "白哈巴", "entity_label": "目的地"}
{"entity_text": "北极村", "entity_label": "返程地"}

文本：冬至去玩半个月
输出：
{"entity_text": "冬至", "entity_label": "时间"}
{"entity_text": "半个月", "entity_label": "时长"}

文本：坐渡轮去度假屋，尝尝烤全羊，买点特产
输出：
{"entity_text": "渡轮", "entity_label": "交通"}
{"entity_text": "度假屋", "entity_label": "住宿"}
{"entity_text": "烤全羊", "entity_label": "餐饮"}
{"entity_text": "特产", "entity_label": "产品"}

文本：预算8800，退休人群轻松游，找小众冷门的地方，三人行，挑晴天去
输出：
{"entity_text": "8800", "entity_label": "预算"}
{"entity_text": "退休", "entity_label": "人群"}
{"entity_text": "轻松游", "entity_label": "强度"}
{"entity_text": "小众冷门", "entity_label": "人气"}
{"entity_text": "三人行", "entity_label": "人数"}
{"entity_text": "晴天", "entity_label": "天气"}"""

# Relation type constraints: for each of the 6 relations, the allowed ob1/ob2 entity_label pairs.
# 1. experience content:
# - ob1 entity_label ∈ [destination, dining, lodging]
# - ob2 entity_label ∈ [product]
# 2. budget limit:
# - ob1 entity_label ∈ [destination, dining, lodging, product, transport]
# - ob2 entity_label ∈ [budget]
# 3. per-capita budget:
# - ob1 entity_label ∈ [budget]
# - ob2 entity_label ∈ [headcount]
# 4. length of stay:
# - ob1 entity_label ∈ [destination, dining, lodging, product]
# - ob2 entity_label ∈ [duration]
# 5. travel time:
# - ob1 entity_label ∈ [departure, return, destination, dining, lodging, product]
# - ob2 entity_label ∈ [time]
# 6. visit order:
# - ① ob1 entity_label ∈ [departure], ob2 entity_label ∈ [destination]
# - ② ob1 entity_label ∈ [destination], ob2 entity_label ∈ [destination]
# - ③ ob1 entity_label ∈ [destination], ob2 entity_label ∈ [return]
PAPER_RELATION_CONSTRAINTS = """1.体验内容：
- ob1 entity_label ∈ [目的地, 餐饮, 住宿]
- ob2 entity_label ∈ [产品]
2.预算限制：
- ob1 entity_label ∈ [目的地, 餐饮, 住宿, 产品, 交通]
- ob2 entity_label ∈ [预算]
3.人均预算：
- ob1 entity_label ∈ [预算]
- ob2 entity_label ∈ [人数]
4.停留时长：
- ob1 entity_label ∈ [目的地, 餐饮, 住宿, 产品]
- ob2 entity_label ∈ [时长]
5.出行时间：
- ob1 entity_label ∈ [出发地, 返程地, 目的地, 餐饮, 住宿, 产品]
- ob2 entity_label ∈ [时间]
6.游玩顺序：
- ① ob1 entity_label ∈ [出发地]，ob2 entity_label ∈ [目的地]
- ② ob1 entity_label ∈ [目的地]，ob2 entity_label ∈ [目的地]
- ③ ob1 entity_label ∈ [目的地]，ob2 entity_label ∈ [返程地]"""

# Entity-aware RE prompt (E branch): includes the entity list and the type constraints.
# You are an expert in Chinese text relation extraction.
# Identify all relation data, with the following relation types: visit order, travel time, length of stay, budget limit, per-capita budget, experience content.
# To extract the personalized intent needs in the user's travel planning.
# Notes:
# {PAPER_RELATION_CONSTRAINTS}  (the relation type constraints defined above)
# All triples must follow the entity-type pairs above;
# All entities must come from the input entities;
# Output format is a JSON list, each element containing three fields: "ob1", "rel", "ob2";
# Output example:
# [
#   {"ob1": "x", "rel": "x", "ob2": "x"},
#   {"ob1": "x", "rel": "x", "ob2": "x"}
# ]
# If there is no relation, output: "no relation found".
PAPER_RE_E_SYSTEM_PROMPT = f"""你是一个中文文本关系抽取领域的专家。
识别所有的关系数据，有如下关系类型：游玩顺序、出行时间、停留时长、预算限制、人均预算、体验内容。
为了提取用户旅游规划中的个性化意图需求。
注释：
{PAPER_RELATION_CONSTRAINTS}
所有三元组必须遵循上述实体类型对；
所有实体必须来自输入的实体；
输出格式为JSON列表，每个元素包含三个字段：\"ob1\"、\"rel\"、\"ob2\"；
输出示例：
[
  {{\"ob1\": \"x\", \"rel\": \"x\", \"ob2\": \"x\"}},
  {{\"ob1\": \"x\", \"rel\": \"x\", \"ob2\": \"x\"}}
]
若无关系则输出：\"找不到关系\"。"""

# Entity-aware hard RE prompt: relation types + type constraints, without the entity list.
# You are an expert in Chinese text relation extraction. Identify all relation data; the relation types can only be: visit order, travel time, length of stay, budget limit, per-capita budget, experience content.
# To extract the personalized intent needs in the user's travel planning.
# {PAPER_RELATION_CONSTRAINTS}
# All triples must follow the entity-type pairs above. Output format is a JSON list, each element containing three fields: "ob1", "rel", "ob2". If there is no relation, output "no relation found".
PAPER_RE_EH_SYSTEM_PROMPT = f"""你是一个中文文本关系抽取领域的专家。识别所有的关系数据，关系类型只能是：游玩顺序、出行时间、停留时长、预算限制、人均预算、体验内容。
为了提取用户旅游规划中的个性化意图需求。
{PAPER_RELATION_CONSTRAINTS}
所有三元组必须遵循上述实体类型对。输出格式为JSON列表，每个元素包含三个字段：\"ob1\"、\"rel\"、\"ob2\"。若无关系则输出\"找不到关系\"。"""

# CDTIR_TO deliberately contains neither an entity list nor entity-type-pair rules.
# You are an expert in Chinese text relation extraction. Identify all relation data; the relation types can only be: visit order, travel time, length of stay, budget limit, per-capita budget, experience content. To extract the personalized intent needs in the user's travel planning. Output format is a JSON list, each element containing three fields: "ob1", "rel", "ob2". If there is no relation, output "no relation found".
PAPER_RE_TO_SYSTEM_PROMPT = """你是一个中文文本关系抽取领域的专家。识别所有的关系数据，关系类型只能是：游玩顺序、出行时间、停留时长、预算限制、人均预算、体验内容。为了提取用户旅游规划中的个性化意图需求。输出格式为JSON列表，每个元素包含三个字段：\"ob1\"、\"rel\"、\"ob2\"。若无关系则输出\"找不到关系\"。"""

# Adds "annotation granularity" few-shot on top of paper_re_e: vague times count as travel time, province->city/subordination counts as visit order.
# Supplementary annotation granularity (must follow):
# 1. travel time: seasons, months, holidays, and vague time periods all count as travel time.
#    Example: request a July travel plan for Shengsi → [{"ob1": "Shengsi", "rel": "travel time", "ob2": "July"}]
#    Example: how to plan a summer route in the Dolomites → [{"ob1": "Dolomites", "rel": "travel time", "ob2": "summer"}]
# 2. visit order: province/country → city/scenic spot, and subordination relations also count as visit order.
#    Example: Yunnan Shangri-La, Lijiang, Dali → [{"ob1": "Yunnan", "rel": "visit order", "ob2": "Shangri-La"}, {"ob1": "Yunnan", "rel": "visit order", "ob2": "Lijiang"}]
PAPER_RE_E_V2_SYSTEM_PROMPT = PAPER_RE_E_SYSTEM_PROMPT + """

补充标注粒度（务必遵守）：
1. 出行时间：季节、月份、假期、模糊时段都算出行时间。
   例：求7月嵊泗旅游规划 → [{"ob1": "嵊泗", "rel": "出行时间", "ob2": "7月"}]
   例：多洛米蒂夏季路线怎么规划 → [{"ob1": "多洛米蒂", "rel": "出行时间", "ob2": "夏季"}]
2. 游玩顺序：省/国家→市/景点、从属关系也算游玩顺序。
   例：云南香格里拉、丽江、大理 → [{"ob1": "云南", "rel": "游玩顺序", "ob2": "香格里拉"}, {"ob1": "云南", "rel": "游玩顺序", "ob2": "丽江"}]"""

# All-category granularity few-shot: gives an example for each of the 6 relations, to avoid hinting only some classes and causing the others to be ignored.
# Supplementary annotation granularity (an example is given for every relation type; must follow):
# 1. visit order (province→city, subordination relations also count):
#    Text: Yunmengzhou Qinglan Mountain, Bibo Lake, Guanlan Temple
#    Output: [{"ob1": "Yunmengzhou", "rel": "visit order", "ob2": "Qinglan Mountain"}, {"ob1": "Yunmengzhou", "rel": "visit order", "ob2": "Bibo Lake"}]
# 2. travel time (seasons, months, holidays, vague time periods also count):
#    Text: Go to Wuyin Peak in late October
#    Output: [{"ob1": "Wuyin Peak", "rel": "travel time", "ob2": "late October"}]
# 3. length of stay (several days/nights, day trip, half a day also count):
#    Text: How to play at Canglan Grassland for 5 days 4 nights
#    Output: [{"ob1": "Canglan Grassland", "rel": "length of stay", "ob2": "5 days 4 nights"}]
# 4. experience content (activities/projects you want to experience also count):
#    Text: Go to Muyun Valley to see the aurora
#    Output: [{"ob1": "Muyun Valley", "rel": "experience content", "ob2": "aurora"}]
# 5. budget limit (amounts, cheap, value-for-money also count):
#    Text: Go to Chengming Lake, budget 3800 yuan
#    Output: [{"ob1": "Chengming Lake", "rel": "budget limit", "ob2": "3800 yuan"}]
# 6. per-capita budget:
#    Text: 4 people go to Tingtao Island, per-capita budget 2600 yuan
#    Output: [{"ob1": "2600 yuan", "rel": "per-capita budget", "ob2": "4"}]
PAPER_RE_E_V3_SYSTEM_PROMPT = PAPER_RE_E_SYSTEM_PROMPT + """

补充标注粒度（每类关系都给出示例，务必遵守）：
1. 游玩顺序（省→市、从属关系也算）：
   文本：云梦州青岚山、碧波湖、观澜寺
   输出：[{"ob1": "云梦州", "rel": "游玩顺序", "ob2": "青岚山"}, {"ob1": "云梦州", "rel": "游玩顺序", "ob2": "碧波湖"}]
2. 出行时间（季节、月份、假期、模糊时段也算）：
   文本：10月下旬去雾隐峰
   输出：[{"ob1": "雾隐峰", "rel": "出行时间", "ob2": "10月下旬"}]
3. 停留时长（几天几晚、一日游、半天也算）：
   文本：沧澜草原5天4晚怎么玩
   输出：[{"ob1": "沧澜草原", "rel": "停留时长", "ob2": "5天4晚"}]
4. 体验内容（想体验的活动、项目也算）：
   文本：去暮云谷想看极光
   输出：[{"ob1": "暮云谷", "rel": "体验内容", "ob2": "极光"}]
5. 预算限制（金额、便宜、性价比也算）：
   文本：去澄明湖玩，预算3800元
   输出：[{"ob1": "澄明湖", "rel": "预算限制", "ob2": "3800元"}]
6. 人均预算：
   文本：4个人去听涛岛，人均预算2600元
   输出：[{"ob1": "2600元", "rel": "人均预算", "ob2": "4"}]"""

# Canonical NER prompt.
# You are a Chinese tourism text entity recognition expert. Please extract entities from the user text; the entity types can only be: destination, departure, return, dining, lodging, product, transport, budget, duration, time, crowd, intensity, popularity, headcount, weather.
# Output only a JSON array, no explanation. Each element has the format {"entity_text":"...","entity_label":"..."}. When there is no entity, output [].
NER_SYSTEM_PROMPT = """你是中文旅游文本实体识别专家。请从用户文本中抽取实体，实体类型只能是：目的地、出发地、返程地、餐饮、住宿、产品、交通、预算、时长、时间、人群、强度、人气、人数、天气。
只输出 JSON 数组，不要解释。每个元素格式为 {\"entity_text\":\"...\",\"entity_label\":\"...\"}。没有实体时输出 []。"""

# Closely follows the original entity_QW2.5*.py / entity_router_rola.py prompt.
# You are an expert in text entity recognition; you need to extract destination; departure; return; dining; lodging; product; transport; budget; duration; time; crowd; intensity; popularity; headcount; weather from the given sentence. Output in json format, e.g. {"entity_text": "Beijing", "entity_label": "destination"} Note: 1. every line of output must be a correct json string. 2. When no entity is found, output "no entity found".
NER_ORIGINAL_SYSTEM_PROMPT = """你是一个文本实体识别领域的专家，你需要从给定的句子中提取 目的地; 出发地; 返程地; 餐饮; 住宿; 产品; 交通; 预算; 时长; 时间; 人群; 强度; 人气; 人数; 天气. 以 json 格式输出, 如 {\"entity_text\": \"北京\", \"entity_label\": \"目的地\"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出\"没有找到任何实体\"."""

# Byte-for-byte content of the triple-quoted system_prompt used by
# entity_QW2.5_fl.py and entity_router_rola.py (leading newline, 4-space indentation, trailing newline+spaces).
# Content translation: You are an expert in text entity recognition; you need to extract destination; departure; return; dining; lodging; product; transport; budget; duration; time; crowd; intensity; popularity; headcount; weather from the given sentence. Output in json format, e.g. {"entity_text": "Beijing", "entity_label": "destination"} Note: 1. every line of output must be a correct json string. 2. When no entity is found, output "no entity found".
NER_ORIGINAL_STRICT_SYSTEM_PROMPT = '\n    你是一个文本实体识别领域的专家，你需要从给定的句子中提取 目的地; 出发地; 返程地; 餐饮; 住宿; 产品; 交通; 预算; 时长; 时间; 人群; 强度; 人气; 人数; 天气. 以 json 格式输出, 如 {"entity_text": "北京", "entity_label": "目的地"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体".\n    '

# Canonical RE prompt.
# You are a Chinese tourism text relation extraction expert. The relation types can only be: visit order, travel time, length of stay, budget limit, per-capita budget, experience content.
# Output only a JSON array, no explanation. Each element has the format {"subject":"...","predicate":"...","object":"..."}. When there is no relation, output [].
# Constraints: experience content=(destination/dining/lodging, product); budget limit=(destination/dining/lodging/product/transport, budget); per-capita budget=(budget, headcount); length of stay=(destination/dining/lodging/product, duration); travel time=(departure/return/destination/dining/lodging/product, time); visit order=(departure->destination, destination->destination, destination->return).
RELATION_SYSTEM_PROMPT = """你是中文旅游文本关系抽取专家。关系类型只能是：游玩顺序、出行时间、停留时长、预算限制、人均预算、体验内容。
请只输出 JSON 数组，不要解释。每个元素格式为 {\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\"}。没有关系时输出 []。
约束：体验内容=(目的地/餐饮/住宿, 产品)；预算限制=(目的地/餐饮/住宿/产品/交通, 预算)；人均预算=(预算, 人数)；停留时长=(目的地/餐饮/住宿/产品, 时长)；出行时间=(出发地/返程地/目的地/餐饮/住宿/产品, 时间)；游玩顺序=(出发地->目的地、目的地->目的地、目的地->返程地)。"""

# Closely follows the tourism NRE prompt used in qwen-t.ipynb/qw_e.ipynb.
# You are an expert in Chinese text relation extraction. Please carefully analyze the following text and strictly follow the requirements below:
# I. Identify all relation data, with the following relation types: visit order, travel time, length of stay, budget limit, per-capita budget, experience content.
# II. The entity type combinations corresponding to the relations are as follows:
# 1. experience content: ob1 entity_label ∈ [destination, dining, lodging]; ob2 entity_label ∈ [product]
# 2. budget limit: ob1 entity_label ∈ [destination, dining, lodging, product, transport]; ob2 entity_label ∈ [budget]
# 3. per-capita budget: ob1 entity_label ∈ [budget]; ob2 entity_label ∈ [headcount]
# 4. length of stay: ob1 entity_label ∈ [destination, dining, lodging, product]; ob2 entity_label ∈ [duration]
# 5. travel time: ob1 entity_label ∈ [departure, return, destination, dining, lodging, product]; ob2 entity_label ∈ [time]
# 6. visit order: departure->destination, destination->destination, destination->return.
# III. All triples must follow the entity-type pairs above.
# IV. Output format is a JSON list, each element containing three fields: "ob1", "rel", "ob2". If there is no relation, output "no relation found".
RELATION_ORIGINAL_SYSTEM_PROMPT = """你是一个中文文本关系抽取领域的专家，请仔细分析以下文本，严格按以下要求执行：
一.识别所有的关系数据,有如下关系类型：游玩顺序,出行时间,停留时长,预算限制,人均预算,体验内容。
二.关系对应的实体类型组合如下：
1.体验内容：ob1 entity_label ∈ [目的地, 餐饮, 住宿]；ob2 entity_label ∈ [产品]
2.预算限制：ob1 entity_label ∈ [目的地, 餐饮, 住宿, 产品, 交通]；ob2 entity_label ∈ [预算]
3.人均预算：ob1 entity_label ∈ [预算]；ob2 entity_label ∈ [人数]
4.停留时长：ob1 entity_label ∈ [目的地, 餐饮, 住宿, 产品]；ob2 entity_label ∈ [时长]
5.出行时间：ob1 entity_label ∈ [出发地, 返程地, 目的地, 餐饮, 住宿, 产品]；ob2 entity_label ∈ [时间]
6.游玩顺序：出发地->目的地、目的地->目的地、目的地->返程地。
三.所有三元组必须遵循上述实体类型对。
四.输出格式为 JSON 列表，每个元素包含三个字段：\"ob1\", \"rel\", \"ob2\"。若无关系则输出\"找不到关系\"。"""

# Entity-aware RE prompt (E branch): includes entity list and constraint.
# Matches train_e.json / qw_e.ipynb from the original project.
# You are an expert in Chinese text relation extraction. Please carefully analyze the following text and strictly follow the requirements below:
# I. Identify all relation data, with the following relation types: visit order, travel time, length of stay, budget limit, per-capita budget, experience content.
# II. The entity type combinations corresponding to the relations are as follows:
# 1. experience content: ob1 entity_label ∈ [destination, dining, lodging]; ob2 entity_label ∈ [product]
# 2. budget limit: ob1 entity_label ∈ [destination, dining, lodging, product, transport]; ob2 entity_label ∈ [budget]
# 3. per-capita budget: ob1 entity_label ∈ [budget]; ob2 entity_label ∈ [headcount]
# 4. length of stay: ob1 entity_label ∈ [destination, dining, lodging, product]; ob2 entity_label ∈ [duration]
# 5. travel time: ob1 entity_label ∈ [departure, return, destination, dining, lodging, product]; ob2 entity_label ∈ [time]
# 6. visit order: departure->destination, destination->destination, destination->return.
# III. All triples must follow the entity-type pairs above.
# IV. All entities must come from the input entities.
# V. Output format is a JSON list, each element containing three fields: "ob1", "rel", "ob2". If there is no relation, output "no relation found".
RELATION_ENTITY_AWARE_SYSTEM_PROMPT = """你是一个中文文本关系抽取领域的专家，请仔细分析以下文本，严格按以下要求执行：
一.识别所有的关系数据,有如下关系类型：游玩顺序,出行时间,停留时长,预算限制,人均预算,体验内容。
二.关系对应的实体类型组合如下：
1.体验内容：ob1 entity_label ∈ [目的地, 餐饮, 住宿]；ob2 entity_label ∈ [产品]
2.预算限制：ob1 entity_label ∈ [目的地, 餐饮, 住宿, 产品, 交通]；ob2 entity_label ∈ [预算]
3.人均预算：ob1 entity_label ∈ [预算]；ob2 entity_label ∈ [人数]
4.停留时长：ob1 entity_label ∈ [目的地, 餐饮, 住宿, 产品]；ob2 entity_label ∈ [时长]
5.出行时间：ob1 entity_label ∈ [出发地, 返程地, 目的地, 餐饮, 住宿, 产品]；ob2 entity_label ∈ [时间]
6.游玩顺序：出发地->目的地、目的地->目的地、目的地->返程地。
三.所有三元组必须遵循上述实体类型对。
四.所有实体必须来自输入的实体。
五.输出格式为 JSON 列表，每个元素包含三个字段：\"ob1\", \"rel\", \"ob2\"。若无关系则输出\"找不到关系\"。"""

# ── Strict RE prompts: byte-level reproduction of original train_t.json / train_e.json ──
# These put the FULL instruction + text into a single user message (no system/user split),
# matching the original qw_e.ipynb / qwen-t.ipynb notebook format exactly.

# You are an expert in Chinese text relation extraction. Please carefully analyze the following text and strictly follow the requirements below:
#
# Requirements:
# I. Identify all relation data, with the following relation types: visit order, travel time, length of stay, budget limit, per-capita budget, experience content.
# II. The entity type combinations corresponding to the relations are as follows:
#     1. experience content:
#     - ob1 entity_label ∈ [destination, dining, lodging]
#     - ob2 entity_label ∈ [product]
#     2. budget limit:
#     - ob1 entity_label ∈ [destination, dining, lodging, product, transport]
#     - ob2 entity_label ∈ [budget]
#     3. per-capita budget:
#     - ob1 entity_label ∈ [budget]
#     - ob2 entity_label ∈ [headcount]
#     4. length of stay:
#     - ob1 entity_label ∈ [destination, dining, lodging, product]
#     - ob2 entity_label ∈ [duration]
#     5. travel time:
#     - ob1 entity_label ∈ [departure, return, destination, dining, lodging, product]
#     - ob2 entity_label ∈ [time]
#     6. visit order:
#     - ① ob1 entity_label ∈ [departure], ob2 entity type ∈ [destination]
#     - ② ob1 entity_label ∈ [destination], ob2 entity type ∈ [destination]
#     - ③ ob1 entity_label ∈ [destination], ob2 entity type ∈ [return]
# III. All triples must follow the entity-type pairs above
# IV. Output format is a JSON list, each element containing three fields: "ob1", "rel", "ob2"
#
# Output example:
# [
#   {"ob1": "x", "rel": "x", 'ob2':'x'},
#   {"ob1": "x", "rel": "x", 'ob2':'x'},
# ]
#
# If there is no relation, output: "no relation found"
#
# Please process the following text:
#
# {text}
RELATION_TEXT_ONLY_STRICT_USER = """你是一个中文文本关系抽取领域的专家，请仔细分析以下文本，严格按以下要求执行：

要求：
一.识别所有的关系数据,有如下关系类型：游玩顺序,出行时间,停留时长,预算限制,人均预算,体验内容。
二.关系对应的实体类型组合如下：
    1. 体验内容：
    - ob1 entity_label ∈ [目的地, 餐饮, 住宿]
    - ob2 entity_label ∈ [产品]
    2. 预算限制：
   - ob1 entity_label ∈ [目的地, 餐饮, 住宿, 产品, 交通]
   - ob2 entity_label ∈ [预算]
    3. 人均预算：
   - ob1 entity_label ∈ [预算]
   - ob2 entity_label ∈ [人数]
    4. 停留时长：
   - ob1 entity_label ∈ [目的地, 餐饮, 住宿, 产品]
   - ob2 entity_label ∈ [时长]
    5. 出行时间：
   - ob1 entity_label ∈ [出发地, 返程地, 目的地, 餐饮, 住宿, 产品]
   - ob2 entity_label ∈ [时间]
    6. 游玩顺序：
   - ① ob1 entity_label ∈ [出发地]，ob2 实体类型 ∈ [目的地]
   - ② ob1 entity_label ∈ [目的地]，ob2 实体类型 ∈ [目的地]
   - ③ ob1 entity_label ∈ [目的地]，ob2 实体类型 ∈ [返程地]
三.所有三元组必须遵循上述实体类型对
四.输出格式为JSON列表，每个元素包含三个字段：\"ob1\", \"rel\", \"ob2\"

输出示例：
[
  {"ob1": "x", "rel": "x", 'ob2':'x'},
  {"ob1": "x", "rel": "x", 'ob2':'x'},
]

若无关系则输出:\"找不到关系\"

请处理以下文本：

{text}"""

# You are an expert in Chinese text relation extraction. Please carefully analyze the following text and strictly follow the requirements below:
#
# Requirements:
# I. Identify all relation data, with the following relation types: visit order, travel time, length of stay, budget limit, per-capita budget, experience content.
# II. The entity type combinations corresponding to the relations are as follows:
#     1. experience content:
#     - ob1 entity_label ∈ [destination, dining, lodging]
#     - ob2 entity_label ∈ [product]
#     2. budget limit:
#     - ob1 entity_label ∈ [destination, dining, lodging, product, transport]
#     - ob2 entity_label ∈ [budget]
#     3. per-capita budget:
#     - ob1 entity_label ∈ [budget]
#     - ob2 entity_label ∈ [headcount]
#     4. length of stay:
#     - ob1 entity_label ∈ [destination, dining, lodging, product]
#     - ob2 entity_label ∈ [duration]
#     5. travel time:
#     - ob1 entity_label ∈ [departure, return, destination, dining, lodging, product]
#     - ob2 entity_label ∈ [time]
#     6. visit order:
#     - ① ob1 entity_label ∈ [departure], ob2 entity type ∈ [destination]
#     - ② ob1 entity_label ∈ [destination], ob2 entity type ∈ [destination]
#     - ③ ob1 entity_label ∈ [destination], ob2 entity type ∈ [return]
# III. All triples must follow the entity-type pairs above
# IV. All entities must come from the input entities
# V. Output format is a JSON list, each element containing three fields: "ob1", "rel", "ob2"
#
# Output example:
# [
#   {"ob1": "x", "rel": "x", 'ob2':'x'},
#   {"ob1": "x", "rel": "x", 'ob2':'x'},
# ]
#
# If there is no relation, output: "no relation found"
#
# Please process the following text, the corresponding entities for the text are given:
#
# {text}
RELATION_ENTITY_AWARE_STRICT_USER = """你是一个中文文本关系抽取领域的专家，请仔细分析以下文本，严格按以下要求执行：

要求：
一.识别所有的关系数据,有如下关系类型：游玩顺序,出行时间,停留时长,预算限制,人均预算,体验内容。
二.关系对应的实体类型组合如下：
    1. 体验内容：
    - ob1 entity_label ∈ [目的地, 餐饮, 住宿]
    - ob2 entity_label ∈ [产品]
    2. 预算限制：
   - ob1 entity_label ∈ [目的地, 餐饮, 住宿, 产品, 交通]
   - ob2 entity_label ∈ [预算]
    3. 人均预算：
   - ob1 entity_label ∈ [预算]
   - ob2 entity_label ∈ [人数]
    4. 停留时长：
   - ob1 entity_label ∈ [目的地, 餐饮, 住宿, 产品]
   - ob2 entity_label ∈ [时长]
    5. 出行时间：
   - ob1 entity_label ∈ [出发地, 返程地, 目的地, 餐饮, 住宿, 产品]
   - ob2 entity_label ∈ [时间]
    6. 游玩顺序：
   - ① ob1 entity_label ∈ [出发地]，ob2 实体类型 ∈ [目的地]
   - ② ob1 entity_label ∈ [目的地]，ob2 实体类型 ∈ [目的地]
   - ③ ob1 entity_label ∈ [目的地]，ob2 实体类型 ∈ [返程地]
三.所有三元组必须遵循上述实体类型对
四.所有实体必须来自输入的实体
五.输出格式为JSON列表，每个元素包含三个字段：\"ob1\", \"rel\", \"ob2\"

输出示例：
[
  {"ob1": "x", "rel": "x", 'ob2':'x'},
  {"ob1": "x", "rel": "x", 'ob2':'x'},
]

若无关系则输出:"找不到关系"

请处理以下文本，已给出文本对应实体：

{text}"""


def system_prompt(task: str, profile: str = "canonical") -> str:
    """[Function description] system_prompt
    - Responsibility: selects the system prompt based on task/profile, clarifying the entity/relation label scope and the output JSON format the model must follow.
    - Main parameters: task: str, profile: str='canonical'.
    - Returns: the return type is annotated as `str`; see the implementation below for field meanings.
    - Note: this function mainly performs in-memory computation; apart from the behavior of the called objects, it has no extra persistence side effects.
    """
    original = profile in {"original", "original_strict"}
    if task == "ner":
        if profile == "paper_ner":
            return PAPER_NER_SYSTEM_PROMPT
        if profile == "paper_ner_fewshot":
            return PAPER_NER_FEWSHOT_SYSTEM_PROMPT
        if profile == "paper_ner_fewshot4":
            return PAPER_NER_FEWSHOT_4GROUP
        if profile == "original_strict":
            return NER_ORIGINAL_STRICT_SYSTEM_PROMPT
        return NER_ORIGINAL_SYSTEM_PROMPT if original else NER_SYSTEM_PROMPT
    # RE task
    if profile == "paper_re_e":
        return PAPER_RE_E_SYSTEM_PROMPT
    if profile == "paper_re_e_v2":
        return PAPER_RE_E_V2_SYSTEM_PROMPT
    if profile == "paper_re_e_v3":
        return PAPER_RE_E_V3_SYSTEM_PROMPT
    if profile == "paper_re_eh":
        return PAPER_RE_EH_SYSTEM_PROMPT
    if profile == "paper_re_to":
        return PAPER_RE_TO_SYSTEM_PROMPT
    if profile == "entity_aware":
        return RELATION_ENTITY_AWARE_SYSTEM_PROMPT
    # text_only_strict / entity_aware_strict: full content goes into user message;
    # system_prompt returns a placeholder that is never shown to the model.
    if profile in ("text_only_strict", "entity_aware_strict"):
        return ""
    return RELATION_ORIGINAL_SYSTEM_PROMPT if original else RELATION_SYSTEM_PROMPT
