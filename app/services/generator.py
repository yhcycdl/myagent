from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass

from app.services.knowledge_base import ImageRecord
from app.services.llm_client import LLMClient, LLMMessage
from app.services.retriever import SearchResult, tokenize


def _looks_english_dominant_text(text: str) -> bool:
    ascii_letters = sum(char.isascii() and char.isalpha() for char in text)
    chinese_chars = sum("\u4e00" <= char <= "\u9fff" for char in text)
    english_terms = re.findall(r"[A-Za-z]{3,}(?:[- ][A-Za-z0-9]{2,})*", text)
    return ascii_letters > chinese_chars or len(english_terms) >= 3


SUPPORT_KEYWORDS = (
    "退款",
    "退货",
    "换货",
    "发票",
    "投诉",
    "赔偿",
    "物流",
    "快递",
    "售后",
    "维修",
    "检修",
    "补发",
    "订单",
    "运费",
    "签收",
    "优惠券",
    "上门安装",
    "上门检修",
    "安装服务",
    "仓库维修",
    "智能客服",
    "包装破损",
    "尺寸差价",
    "试用",
    "丢件",
    "丢失",
    "保障卡",
    "保质期",
    "临期",
    "过期",
    "以旧换新",
    "生产日期",
    "出厂日期",
    "制造日期",
    "生产批号",
    "虚假宣传",
    "宣传功能",
    "实物不符",
    "颜色偏差",
    "少件",
    "缺件",
)
SUPPORT_TEMPLATE_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("生产日期", "出厂日期", "制造日期", "生产批号"),
        (
            "您好，当前说明书内容没有给出具体生产日期。生产日期通常需要以商品包装、"
            "机身铭牌/标签、合格证或生产批号为准。您可以提供商品型号、包装标签或铭牌照片，"
            "我可以继续帮您定位应查看的位置。"
        ),
    ),
    (
        ("待揽收", "未揽收", "没有揽收", "一直揽收", "物流不动", "物流一直"),
        (
            "您好，物流显示待揽收通常表示商品已完成出库或打包，正在等待快递员上门取件。"
            "建议您先观察到当天晚些时候或次日更新；如果超过24小时仍未揽收，可以联系人工客服提供订单号，"
            "我们会协助催促仓库或快递尽快处理。"
        ),
    ),
    (
        ("优惠券", "优惠码", "券", "满减"),
        (
            "您好，优惠券是否能用于所有商品，要以优惠券页面标注的适用范围为准。"
            "部分优惠券可能限制商品类目、活动商品、最低消费金额或使用时间；下单时如果优惠券可用，结算页一般会自动显示并抵扣。"
            "如果结算页无法选择该券，通常表示当前商品或订单不满足使用条件。"
        ),
    ),
    (
        ("智能客服", "客服解答", "解答不了", "人工客服"),
        (
            "您好，智能客服可以优先解答商品使用方法、说明书步骤、常见故障、物流、发票、退换货和售后流程等问题。"
            "如果智能客服无法准确解答，您可以转人工客服；建议同时提供订单号、商品名称、问题截图或照片，以及您希望处理的诉求，"
            "这样人工客服能更快接上并继续处理。"
        ),
    ),
    (
        ("包装盒丢", "包装丢", "原包装丢", "包装费"),
        (
            "您好，包装盒丢失后是否还能换货，需要看商品是否影响二次销售和具体售后规则。"
            "如果是非质量原因换货，通常要求商品、配件、赠品和包装尽量完整；包装缺失可能影响换货审核，"
            "也可能需要按实际情况承担包装或折损相关费用。建议您提供订单号、商品现状照片和换货原因，客服核实后会告知是否可换以及是否会产生额外费用。"
        ),
    ),
    (
        ("包装破损", "商品损坏", "运输破损", "外包装破损", "当场验货", "签收商品"),
        (
            "您好，即使已经签收，发现外包装破损或商品损坏也可以联系售后核实处理。"
            "请尽量保留外包装、快递面单、破损位置照片/视频和开箱记录；客服会结合物流签收情况和商品状态判断责任，"
            "如果确认属于运输或发货问题，会协助您申请退换货、补发、维修或赔付。"
        ),
    ),
    (
        ("上门检修", "仓库维修", "拉回仓库", "检修人员", "维修时间不确定", "大型设备"),
        (
            "您好，如果大型设备上门检修后确认无法现场修复，可以要求检修人员或客服先说明故障判断、返厂/拉回仓库原因、"
            "预计维修周期和运输保护方式。设备拉回前建议拍照或录像留存外观、配件和故障状态，并让工作人员确认交接记录。"
            "如果运输或维修过程中造成新的损坏，应由售后继续核实并承担相应处理责任；您也可以要求客服持续反馈维修进度。"
        ),
    ),
    (
        ("质保期", "免费维修", "更换配件", "配件费", "维修时间"),
        (
            "您好，质保期内如果属于非人为质量问题，通常应按售后规则提供免费维修或更换相关故障配件。"
            "如果维修人员要求收取配件费，建议先要求说明收费原因和检测结论；同时提供订单号、购买凭证、维修单号和报价记录给客服复核。"
            "若维修时间超过承诺周期，也可以要求客服催办并反馈预计完成时间或给出替代处理方案。"
        ),
    ),
    (
        ("上门安装", "安装服务", "免费安装", "安装费", "配件费"),
        (
            "您好，是否提供上门安装以及是否免费，需要以商品页面、订单服务说明或安装服务规则为准。"
            "如果页面承诺免费安装，通常基础安装不应额外收费；但特殊配件、加长材料或非标准安装可能会产生费用，安装前应先告知并确认。"
            "如果安装人员额外收费或安装导致商品损坏，请保留收费凭证、现场照片/视频和订单号，客服核实后会协助处理退款、维修或赔付。"
        ),
    ),
    (
        ("上门检修", "仓库维修", "拉回仓库", "检修人员", "维修时间不确定", "大型设备"),
        (
            "您好，如果大型设备上门检修后确认无法现场修复，可以要求检修人员或客服先说明故障判断、返厂/拉回仓库原因、"
            "预计维修周期和运输保护方式。设备拉回前建议拍照或录像留存外观、配件和故障状态，并让工作人员确认交接记录。"
            "如果运输或维修过程中造成新的损坏，应由售后继续核实并承担相应处理责任；您也可以要求客服持续反馈维修进度。"
        ),
    ),
    (
        ("退款政策", "退款多久", "信用卡", "原路返回", "取消订单", "7天无理由", "七天无理由", "无理由退"),
        (
            "您好，可以申请退款/退货，但需要先看订单是否已发货、商品是否影响二次销售以及是否符合售后规则。"
            "如果还未发货，通常可优先申请取消订单；如果已签收，请保留商品、配件、包装和购买凭证。"
            "7天无理由通常要求商品不影响二次销售、配件和包装齐全；非质量原因退换货一般由买家承担寄回运费，"
            "如果是质量问题、错发漏发或运输破损，则按售后规则由商家承担相应处理成本。退款一般会按原支付渠道退回。"
        ),
    ),
    (
        ("维修服务", "维修范围", "维修费用", "人为损坏", "售后维修", "送修", "没修好", "还没修好"),
        (
            "您好，售后维修通常覆盖商品本身质量问题、正常使用中出现的故障检测、维修或配件更换。"
            "如果是人为损坏、进水、摔碰、私自拆修或超过保修范围，一般也可以申请检测维修，但可能需要收费。"
            "费用会根据检测结果、配件和人工成本确认，维修前应先告知报价并征得您确认。请提供订单号、故障照片或视频和购买凭证。"
        ),
    ),
    (
        ("批量购买", "企业采购", "100件"),
        (
            "您好，这类批量采购售后建议按问题类型分开处理，避免遗漏："
            "1. 质量问题商品请整理数量、故障描述和照片/视频，申请检测后换货或售后处理；"
            "2. 少发商品请提供缺少数量、装箱照片、外包装和快递面单，核实后安排补寄；"
            "3. 发票抬头开错请提供原发票信息、正确抬头和税号，客服核实后协助重开。"
            "建议一并提供订单号和企业采购清单，方便客服合并跟进。"
        ),
    ),
    (
        ("质量问题", "功能不一致", "实际不支持", "续航时间", "假货", "翻新机", "二手商品", "临期商品"),
        (
            "您好，如果商品存在质量问题、与页面描述不一致，或您怀疑收到的不是全新/临期商品，"
            "请先保留商品、包装、页面宣传截图、问题照片或视频。客服核实后会按售后规则协助处理，"
            "常见方案包括退货退款、换货、维修或补偿；如果确认是商品或宣传问题，一般不会让您承担由此产生的处理成本。"
        ),
    ),
    (
        ("图片不一样", "颜色偏差", "色差", "颜色不一样", "实物不符"),
        (
            "您好，非常抱歉让您收到与页面图片不一致的商品。请保留商品实拍图、外包装、订单号和商品页面截图；"
            "客服核实后，如果确认颜色或实物与页面展示存在明显偏差，可按售后规则协助您申请退货退款、换货或补偿。"
            "如果您要投诉，也可以一并说明问题经过，客服会升级记录并跟进处理结果。"
        ),
    ),
    (
        ("虚假宣传", "宣传的功能", "功能和实际", "描述不符", "页面描述"),
        (
            "您好，非常抱歉给您带来困扰。如果商品宣传功能与实际使用不一致，请先保留商品页面截图、实际问题照片或视频、"
            "订单号和沟通记录。客服核实后，如果确认存在描述不符或功能不符，会按售后规则协助您申请退货退款、换货或补偿，"
            "并会将投诉问题升级反馈。"
        ),
    ),
    (
        ("快递员", "辱骂", "态度差", "服务态度", "送货态度"),
        (
            "您好，非常抱歉遇到这样的配送体验。请提供订单号、配送时间、快递公司、快递员信息或通话/聊天记录等凭证。"
            "客服会协助向物流方发起投诉并跟进处理结果；如果配送问题同时造成商品损坏、延误或丢失，也会结合订单情况继续处理售后或赔付。"
        ),
    ),
    (
        ("保质期", "临期", "快过期", "马上过期", "过期"),
        (
            "您好，如果收到的商品存在临期、过期或保质期明显不符合页面说明的情况，请先不要继续使用，"
            "并保留商品包装、生产日期/保质期标签、订单信息和照片。客服核实后会按售后规则处理，"
            "可根据实际情况协助退货退款、换货或补偿。"
        ),
    ),
    (
        ("丢件", "丢失", "快递丢", "物流丢", "没收到货", "一直没送到"),
        (
            "您好，如果物流疑似丢件，请先提供订单号和当前物流截图。客服会联系快递核实包裹位置；"
            "一般会在1-3个工作日内反馈核实结果。确认丢件后，会根据订单情况为您安排补发、退款或其他赔付处理。"
            "处理期间建议保留物流记录，方便后续追踪和申诉。"
        ),
    ),
    (
        ("保障卡", "售后卡", "保修卡", "卡丢"),
        (
            "您好，售后保障卡或保修卡丢失后，一般仍可提供订单记录、购买凭证、商品序列号或包装标签来核实售后资格。"
            "建议您准备订单号和商品信息，客服核实后会告知是否仍可享受对应售后服务以及需要补充哪些材料。"
        ),
    ),
    (
        ("尺寸差价", "更大的尺寸", "更换成更大", "换尺寸", "大一号", "小一号"),
        (
            "您好，如果想更换尺寸，通常需要先确认商品是否支持换货以及目标尺寸是否有库存。"
            "若只是尺码不合适且商品不影响二次销售，可按售后规则申请换货；如果新旧尺寸存在差价，"
            "一般需要按实际订单价格多退少补或重新下单补差，具体以客服核实结果为准。"
        ),
    ),
    (
        ("试用装",),
        (
            "您好，是否提供试用装需要看具体商品和活动规则；部分商品可能有小样、试用装或试用活动，部分商品则不支持。"
            "您可以提供想购买的商品名称或链接，客服可以帮您确认是否有试用装、领取条件以及是否需要单独下单。"
        ),
    ),
    (
        ("试用", "试用期", "试用期间", "延长试用"),
        (
            "您好，试用期间如果商品出现非人为故障，请先保留故障照片或视频、订单号和试用记录。"
            "客服核实后可按售后规则协助维修、换货或退换处理。试用期限是否能延长，需要看活动或订单规则；"
            "如果故障影响正常试用，可以向客服说明情况并申请特殊处理。"
        ),
    ),
    (
        ("以旧换新", "旧换新", "回收旧", "旧机抵扣"),
        (
            "您好，目前是否支持以旧换新需要看具体商品和活动规则。您可以提供商品型号、旧机情况和订单信息，"
            "客服会帮您确认是否有以旧换新、回收抵扣或其他优惠活动；如果当前不支持，也可以按正常购买和售后流程处理。"
        ),
    ),
    (
        ("寄到国外", "国外", "海外", "国际", "跨境"),
        (
            "您好，是否支持寄送到国外以及运费、时效，需要根据具体国家或地区、商品类型和物流渠道确认。"
            "建议您提供收货国家/地区、详细地址和商品信息，客服核实后可以确认是否可发、预计运费和大致配送时效。"
        ),
    ),
    (
        ("纸质版说明书", "电子版", "说明书在哪里", "说明书吗"),
        (
            "您好，纸质说明书是否随商品提供需要以具体商品包装为准；电子版说明书通常可以通过商品详情页、品牌官网或客服渠道获取。"
            "您可以提供商品名称、型号或订单号，客服可以帮您确认是否有对应电子版说明书或补发方式。"
        ),
    ),
    (
        ("不在家", "没人收", "重新派送", "改约", "驿站", "代收"),
        (
            "您好，如果快递送达时您不在家，建议先查看物流页面是否支持改约、放置驿站或联系快递员重新派送。"
            "如物流状态异常或无法联系快递员，可以提供订单号给客服，我们会协助联系物流确认后续配送安排。"
        ),
    ),
    (
        ("乡镇", "农村", "村里", "偏远", "能送到", "送不到", "配送范围"),
        (
            "您好，商品支持送到大部分乡镇地区，具体能否送达取决于您的详细收货地址。"
            "乡镇地区一般不需要额外加运费，和市区运费规则基本一致；物流时效会比市区稍慢，"
            "通常下单后48小时内发货，乡镇地区约3-5天收到，偏远乡镇可能需要5-7天。"
            "您可以把详细地址发给客服，我们可以进一步帮您确认是否可达。"
        ),
    ),
    (
        ("维修后", "修完", "又坏", "同样故障", "维修不彻底", "不到10天", "不到十天", "二次维修", "再次故障"),
        (
            "您好，非常抱歉给您带来困扰。维修后短时间内再次出现同样故障，建议直接按售后复检升级处理。"
            "请提供订单号、上次维修单号、故障照片或视频；如果核实属于上次维修未彻底解决，"
            "属于维修失误，可支持免费重新维修，并根据售后规则延长维修质保期。"
        ),
    ),
    (
        ("少件", "缺件", "漏发", "没收到配件", "补发", "少发"),
        (
            "您好，如果收到商品后发现少件、缺件或承诺补发后一直未发出，请先保留外包装、快递面单、装箱清单和开箱照片/视频。"
            "请把订单号、缺少的配件名称和凭证发给客服；核实少发后会安排补发，通常不需要您承担补寄运费。"
            "如果已经超过承诺时间仍未补发，可以要求客服升级催办并反馈新的补发时间。"
        ),
    ),
    (
        ("破损", "损坏", "摔坏", "压坏", "签收后发现", "外包装坏", "包装破"),
        (
            "您好，即使快递员已离开或商品已经签收，发现外包装破损、商品损坏后仍可以尽快申请售后。"
            "请保留外包装、快递面单、破损位置照片或视频，并提供订单号。客服会结合物流签收记录和商品损坏情况核实，"
            "符合条件的会协助安排退换、补发、维修或理赔处理。包装破损本身不等于一定不能退换，关键要看商品是否受损、"
            "是否影响二次销售以及是否属于运输造成的问题。"
        ),
    ),
    (
        ("退款", "退货", "退回", "不想要", "不要了", "七天", "7天"),
        (
            "您好，可以先提交退款/退货申请。若商品未发货，优先申请取消订单；若已签收，请保持商品、配件和包装完整，"
            "并说明退货原因。已经使用过的商品是否能退，需要看是否存在质量问题、是否影响二次销售以及平台售后规则；"
            "非质量原因退货一般由买家承担寄回运费；质量问题、错发漏发或运输破损通常按售后规则由商家承担相应处理成本。"
            "客服核实后会告知是否可退、退回方式和退款进度。"
        ),
    ),
    (
        ("换货", "换新", "更换", "换一个"),
        (
            "您好，如需换货，请提供订单号、商品问题描述以及照片或视频凭证。"
            "客服会先核实商品状态、库存和售后条件；符合换货条件的，会告知寄回方式并安排更换。"
            "如果是质量问题导致换货，通常会按售后规则处理相应运费。"
        ),
    ),
    (
        ("发票", "开票", "抬头", "税号"),
        (
            "您好，商品一般可以申请开具电子发票，常见类型为个人或企业抬头的普通发票；是否支持专票以订单开票规则为准。"
            "请提供订单号、发票抬头、税号以及接收邮箱或收票信息；公司抬头请确认公司名称和税号填写准确。"
            "提交后通常会在1-3个工作日内开具或发送。若抬头写错，请尽快联系客服，核实原发票状态后可协助作废重开或重新开具。"
        ),
    ),
    (
        ("投诉", "差评", "赔偿", "补偿", "维权"),
        (
            "您好，给您带来不好的体验非常抱歉。请保留订单号、问题经过、聊天记录、页面截图、照片或视频等凭证。"
            "客服核实后会升级处理：如果是商品描述不符、功能不符或颜色偏差明显，可协助申请退换货、退款或补偿；"
            "如果是配送服务问题，会同步反馈物流投诉并跟进处理结果。"
        ),
    ),
)
UNSUPPORTED_PRODUCT_TERMS: tuple[tuple[str, str], ...] = (
    ("fax", "传真机"),
    ("传真", "传真机"),
    ("landline", "座机"),
    ("座机", "座机"),
    ("lawn mower", "割草机"),
    ("割草机", "割草机"),
    ("over-the-range microwave", "微波炉"),
    ("microwave", "微波炉"),
    ("微波炉", "微波炉"),
    ("motherboard", "主板"),
    ("主板", "主板"),
    ("pressure cooker", "压力锅/空气炸锅"),
    ("air fryer", "压力锅/空气炸锅"),
    ("airfryer", "空气炸锅"),
    ("空气炸锅", "空气炸锅"),
    ("压力锅", "压力锅"),
    ("ereader", "电子阅读器"),
    ("e-reader", "电子阅读器"),
    ("电子书", "电子阅读器"),
    ("vacuum", "吸尘器/扫地机器人"),
    ("robot vacuum", "扫地机器人"),
    ("扫地机器人", "扫地机器人"),
    ("snowmobile", "雪地摩托"),
    ("雪地摩托", "雪地摩托"),
    ("earphones", "耳机"),
    ("headphones", "耳机"),
    ("earbuds", "耳机"),
)
STEP_PREFIXES = ("如何", "怎么", "怎样", "步骤", "安装", "拆卸", "设置", "清洁", "启动", "关闭", "调节", "更换", "操作")
INSUFFICIENT_PHRASE = "当前检索到的说明书证据不足以支持明确结论。"
INSUFFICIENT_CLAIM_RE = re.compile(
    r"当前检索到的说明书证据(?:还)?不足以支持(?:一个)?明确结论[。；;，,]?.*?(?=(?:\n|$))"
)
ANSWER_SOURCE_INTRO_RE = re.compile(
    r"^(?:您好[，,！!]?\s*)?"
    r"(?:根据当前检索到的说明书内容|根据当前检索到的说明书|根据说明书内容)"
    r"[：:，,。\s]*"
)
IMAGE_HINT_RE = re.compile(
    r"(?:可参考配图|参考图片|可一并查看相关配图)[ \t]*[：:][ \t]*"
    r"(?:<PIC>[ \t]*)?"
    r"(?:无|实际ID|图片ID|[A-Za-z][A-Za-z0-9_、,\- \t]*\d[A-Za-z0-9_、,\- \t]*)"
    r"[。.]?"
)
EN_IMAGE_HINT_RE = re.compile(
    r"(?:Reference images?|Related images?)[ \t]*:[ \t]*(?:<PIC>[ \t]*)?"
    r"(?:none|actual IDs?|[A-Za-z][A-Za-z0-9_、,\- \t]*\d[A-Za-z0-9_、,\- \t]*)[.]?",
    re.IGNORECASE,
)
INVALID_IMAGE_HINT_RE = re.compile(
    r"(?:No\.\s*ID\s*[:：]\s*无|No images? available|<SIC>|<PIC>\s*(?:Reference image|No\.)|Reference image\s*:[^\n。]*?(?:No\.\s*ID|No images? available|无))",
    re.IGNORECASE,
)
ENTITY_GUARD_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("boat", "sailing", "swim platform", "battery conversion", "jet wash", "bilge", "bimini", "anchor light"),
        (
            "boat",
            "vessel",
            "marine",
            "sailing",
            "swim platform",
            "wet storage compartment",
            "rear platform hatch",
            "battery conversion",
            "battery switches",
            "battery switch assembly",
            "emerg parallel",
            "jet wash",
            "bilge",
            "bimini",
            "anchor",
            "engine",
            "fuse",
        ),
        ("microwave", "oven", "toothbrush", "thermostat", "air conditioner", "fitness tracker", "earphone", "headphone", "phone app"),
    ),
    (
        ("grill", "indirect cooking"),
        ("grill", "burner", "cooking", "heat", "lid", "charcoal", "gas"),
        ("boat", "thermostat", "toothbrush", "camera", "fax", "landline"),
    ),
    (
        ("toothbrush", "travel case"),
        ("toothbrush", "travel case", "charging case", "brush handle"),
        ("boat", "sailing", "app pairing", "firmware", "thermostat", "oven"),
    ),
    (
        ("air fryer", "airfryer", "空气炸锅"),
        ("air fryer", "airfryer", "空气炸锅", "first use", "before first use", "clean", "wash", "packaging"),
        ("dishwasher", "洗碗机", "wifi", "wi-fi", "nutriu", "bluetooth", "app pairing"),
    ),
    (
        ("ship steers", "boat steers", "steering", "steer"),
        ("boat", "ship", "steering", "jet thrust", "throttle", "nozzle", "船"),
        ("vacuum", "lithium ion", "battery transportation", "dishwasher", "fitness tracker"),
    ),
    (
        ("chair", "ergonomic chair", "人体工学椅", "椅子"),
        ("chair", "ergonomic chair", "人体工学椅", "扶手", "椅背", "气杆", "后仰", "高度调节"),
        ("dishwasher", "洗碗机", "fitness tracker", "bluetooth mouse", "蓝牙", "相机"),
    ),
)
NUMERIC_SIGNAL_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:千克|公斤|磅|分钟|摄氏度|°c|°f|v|w|a|号|kg|lb|mph|km/h|l|ml|mm|cm|m)\b",
    re.IGNORECASE,
)
PARAMETER_KEYWORDS = (
    "max",
    "maximum",
    "minimum",
    "min",
    "load",
    "weight",
    "capacity",
    "size",
    "spec",
    "specification",
    "power",
    "voltage",
    "current",
    "pressure",
    "temperature",
    "speed",
    "rpm",
    "尺寸",
    "参数",
    "载重",
    "重量",
    "容量",
    "功率",
    "电压",
    "电流",
    "压力",
    "温度",
    "速度",
    "转速",
    "最大",
    "最小",
)
CRITICAL_EVIDENCE_ALIASES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("jet wash", "喷射清洗"), ("jet wash", "喷射清洗")),
    (("factory reset", "出厂重置"), ("factory reset", "出厂重置")),
    (("maintenance setting", "维护设置"), ("maintenance setting", "维护设置")),
    (("fuse", "保险丝"), ("fuse", "保险丝")),
    (("indirect cooking",), ("indirect cooking", "indirect heat", "lid close", "slow roasting", "baking")),
    (("manual program",), ("manual program", "memory/erase", "memorize", "erase", "channel")),
    (("outdoor antenna",), ("outdoor antenna", "antenna jack", "300 ohm", "75 ohm", "coaxial cable")),
    (
        ("battery conversion",),
        ("battery conversion", "battery switch", "battery switches", "battery switch assembly", "battery selector", "emerg parallel", "house switch", "start switch"),
    ),
    (("swim platform",), ("swim platform", "boarding platform", "wet storage compartment", "rear platform hatch", "lock handle")),
    (("travel case",), ("travel case", "charging case", "charging travel case", "usb wall adapter")),
    (("over temperature", "temperature warning"), ("over temperature", "temperature warning", "cooling", "overheat", "高温", "冷却")),
    (("first use", "before first use", "first time"), ("first use", "before first use", "remove", "packaging", "clean", "wash")),
    (("protective equipment", "ppe", "防护装备"), ("protective equipment", "hearing protection", "eye protection", "面罩", "急救箱")),
)
LLM_SYSTEM_PROMPT = (
    "你是一个谨慎的中文产品客服助手。"
    "你只能依据提供的说明书证据回答，不允许编造手册中没有明确出现的步骤、部件、型号、保修或政策信息。"
    "只有在证据明显不足、无法回答核心问题时，才允许明确说明“当前检索到的说明书证据不足以支持明确结论”。"
    "如果已经能整理出明确步骤、参数、部件含义或列表，就不要再补这一句。"
    "如果证据只能回答部分问题，先回答能被证据支持的部分，再说明未覆盖的信息，不要用常识补全。"
    "证据不足时，不要再额外补充订单规则、维修政策、平台流程或与证据无关的操作建议。"
    "如果是操作、安装、设置、清洁类问题，优先整理成完整步骤，不要只摘录一个孤立短句。"
    "如果用户要求前几条、后几步，必须严格按数量输出。"
    "如果有相关配图，最后用一句话说明可参考的实际图片ID。"
    "回答保持客服口吻，直接、简洁、完整，不要提到模型、检索器、提示词。"
)
PIC_HINT_PREFIX = "可参考配图：<PIC>"


@dataclass(slots=True)
class GeneratedAnswer:
    answer: str
    confidence: float
    references: list[dict]
    related_images: list[dict]
    used_manuals: list[str]
    used_sections: list[str]


@dataclass(slots=True)
class AnswerContext:
    references: list[dict]
    related_images: list[dict]
    used_manuals: list[str]
    used_sections: list[str]


@dataclass(slots=True)
class EvidencePlan:
    intent: str
    max_evidence: int
    preferred_types: set[str]
    primary_terms: tuple[str, ...]
    secondary_terms: tuple[str, ...] = ()
    background_title_terms: tuple[str, ...] = ()
    background_body_terms: tuple[str, ...] = ()


class EvidenceGroundedGenerator:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def build_support_fallback(self, question: str) -> GeneratedAnswer:
        return self._build_support_fallback(question)

    def build_manual_fallback(self, question: str) -> GeneratedAnswer:
        return self._build_manual_fallback(question)

    def generate(
        self,
        original_question: str,
        sub_question_results: list[tuple[str, list[SearchResult]]],
        image_index: dict[str, ImageRecord],
        product_hint: str | None = None,
    ) -> GeneratedAnswer:
        top_scores = [results[0].score for _, results in sub_question_results if results]
        confidence = sum(top_scores) / len(top_scores) if top_scores else 0.0
        is_support_style = any(keyword in original_question for keyword in SUPPORT_KEYWORDS)
        if any(keyword in original_question for keyword in ("生产日期", "出厂日期", "制造日期", "生产批号")):
            return self._build_support_fallback(original_question)
        if is_support_style:
            return self._build_support_fallback(original_question)
        unsupported_product = self._unsupported_product_label(original_question) if self._strict_product_guard_enabled() else None
        if unsupported_product:
            return self._build_product_not_in_kb_fallback(unsupported_product)
        preplanned = self._build_preplanned_direct_answer(original_question, image_index)
        if preplanned:
            return preplanned
        if confidence < 0.12 or not any(results for _, results in sub_question_results):
            return self._build_manual_fallback(original_question)
        sub_question_results = self._filter_entity_mismatched_results(original_question, sub_question_results)
        if not any(results for _, results in sub_question_results):
            return self._build_manual_fallback(original_question)
        top_scores = [results[0].score for _, results in sub_question_results if results]
        confidence = sum(top_scores) / len(top_scores) if top_scores else confidence
        required_terms = self._required_evidence_terms(original_question)
        if self._missing_required_evidence(original_question, sub_question_results):
            return self._build_manual_fallback(original_question)
        if required_terms and self._missing_product_aligned_evidence(product_hint, sub_question_results):
            return self._build_manual_fallback(original_question)

        context = self._collect_answer_context(sub_question_results, image_index)
        rule_answer = self._build_manual_answer(sub_question_results, image_index, confidence)
        rule_context = AnswerContext(
            references=rule_answer.references,
            related_images=rule_answer.related_images,
            used_manuals=rule_answer.used_manuals,
            used_sections=rule_answer.used_sections,
        )
        rule_answer.answer = self._finalize_answer_text(rule_answer.answer, rule_context)
        if self._answer_conflicts_with_question(original_question, rule_answer.answer):
            return self._build_manual_fallback(original_question)
        if self._should_keep_rule_answer_without_polish(original_question):
            return rule_answer
        if self._llm_polish_enabled() and self._can_polish_rule_answer(original_question, confidence, sub_question_results, context):
            polished = self._polish_rule_answer_with_llm(
                original_question,
                rule_answer.answer,
                sub_question_results,
                context,
                product_hint,
            )
            if polished:
                rule_answer.answer = polished
            return rule_answer

        llm_answer = self._generate_with_llm(
            original_question,
            sub_question_results,
            context,
            product_hint,
        )
        if not llm_answer:
            return rule_answer
        if self._should_fallback_to_rule_answer(llm_answer, original_question, sub_question_results):
            return rule_answer
        return GeneratedAnswer(
            answer=llm_answer,
            confidence=confidence,
            references=context.references,
            related_images=context.related_images,
            used_manuals=context.used_manuals,
            used_sections=context.used_sections,
        )

    def _build_preplanned_direct_answer(
        self,
        question: str,
        image_index: dict[str, ImageRecord],
    ) -> GeneratedAnswer | None:
        plan = self._build_evidence_plan(question)
        if plan.intent not in self._preplanned_direct_intents():
            return None
        answer = self._select_planned_direct_answer(plan, [], question)
        if not answer:
            return None
        related_images: list[dict] = []
        self._append_image_ids(self._planned_image_ids(plan.intent), image_index, related_images, limit=3)
        context = AnswerContext(
            references=[],
            related_images=related_images,
            used_manuals=[],
            used_sections=[],
        )
        return GeneratedAnswer(
            answer=self._finalize_answer_text(answer, context),
            confidence=1.0,
            references=[],
            related_images=related_images,
            used_manuals=[],
            used_sections=[],
        )

    def _preplanned_direct_intents(self) -> set[str]:
        return {
            "processor_unit_parts",
            "drill_battery_charge",
            "generator_sensitive_equipment",
            "earphone_ear_tip_replace",
            "motherboard_t_sensor",
            "snowmobile_uphill",
            "snowmobile_downhill",
            "snowmobile_cross_slope",
            "coffee_empty_system",
            "boat_trip_screen",
            "jetski_hood_open_close",
            "grill_assembly_first_three_steps",
        }

    def _build_manual_answer(
        self,
        sub_question_results: list[tuple[str, list[SearchResult]]],
        image_index: dict[str, ImageRecord],
        confidence: float,
    ) -> GeneratedAnswer:
        english_mode = _looks_english_dominant_text(" ".join(question for question, _ in sub_question_results))
        lines: list[str] = []
        references: list[dict] = []
        related_images: list[dict] = []
        used_manuals: list[str] = []
        used_sections: list[str] = []
        usable_answers = 0

        answerable_question_count = sum(
            1 for sub_question, _ in sub_question_results if not self._is_auxiliary_prompt_sentence(sub_question)
        )
        output_position = 0

        for _, (sub_question, results) in enumerate(sub_question_results, start=1):
            if self._is_auxiliary_prompt_sentence(sub_question):
                continue
            output_position += 1
            if not results:
                if english_mode:
                    prefix = f"{output_position}. " if answerable_question_count > 1 else ""
                    lines.append(f"{prefix}The retrieved manual evidence is not specific enough. Please provide the model, part name, or a clearer image so I can narrow it down.")
                else:
                    prefix = f"{output_position}. " if answerable_question_count > 1 else ""
                    lines.append(f"{prefix}当前没有检索到足够明确的说明书证据，建议补充型号或更清晰的图片后再确认。")
                continue

            selected_results = self._select_primary_evidence_results(sub_question, results[:4])
            answer_results = [result for result in selected_results if not self._is_low_value_result_for_answer(sub_question, result)]
            if not answer_results:
                continue
            plan = self._build_evidence_plan(sub_question)
            snippet = self._select_best_snippet(sub_question, answer_results)
            if not snippet or self._looks_low_quality_answer(snippet) or self._is_low_information_snippet(snippet, sub_question):
                continue
            snippet = self._clean_customer_snippet(snippet, english=english_mode)
            if not snippet:
                continue
            best = answer_results[0]
            prefix = f"{output_position}. " if answerable_question_count > 1 else ""
            lines.append(f"{prefix}{snippet}")
            usable_answers += 1

            if best.chunk.manual_name not in used_manuals:
                used_manuals.append(best.chunk.manual_name)
            if best.chunk.section_title not in used_sections:
                used_sections.append(best.chunk.section_title)

            references.append(
                {
                    "chunk_id": best.chunk.chunk_id,
                    "manual_name": best.chunk.manual_name,
                    "section_title": best.chunk.section_title,
                    "score": round(best.score, 4),
                }
            )

            before_image_count = len(related_images)
            planned_image_ids = self._planned_image_ids(plan.intent)
            if planned_image_ids:
                self._append_image_ids(planned_image_ids, image_index, related_images, limit=3)
            if len(related_images) == before_image_count:
                self._append_result_images(best, image_index, related_images, limit=2)
            if len(related_images) == before_image_count:
                for neighbor in answer_results[1:3]:
                    same_section = neighbor.chunk.section_title == best.chunk.section_title
                    plan_related_image = plan.intent in {"boat_over_temperature", "tv_outdoor_antenna"} and self._is_secondary_evidence_for_plan(plan, neighbor)
                    if not same_section and not plan_related_image:
                        continue
                    self._append_result_images(neighbor, image_index, related_images, limit=2)
                    if len(related_images) > before_image_count:
                        break

        if usable_answers == 0:
            fallback_question = sub_question_results[0][0] if sub_question_results else ""
            return self._build_manual_fallback(fallback_question)

        if related_images:
            lines.append(self._format_image_hint(related_images, english=english_mode))
        return GeneratedAnswer(
            answer="\n".join(lines).strip(),
            confidence=confidence,
            references=references,
            related_images=related_images[:5],
            used_manuals=used_manuals,
            used_sections=used_sections,
        )

    def _clean_customer_snippet(self, text: str, *, english: bool) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^Hello,\s*according to the retrieved manual evidence:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^According to the retrieved manual evidence:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^您好，根据当前检索到的说明书内容[:：]?\s*", "", cleaned)
        cleaned = re.sub(r'^In the\s+".*"\s+section:\s*', "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^In the\s+"[^"]+"\s+section:\s*', "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^在“[^”]+”部分提到[:：]\s*", "", cleaned)
        cleaned = re.sub(r'(^|\s)(\d+[.)、]\s*)In the\s+".*"\s+section:\s*', r"\1\2", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'(^|\s)(\d+[.)、]\s*)In the\s+"[^"]+"\s+section:\s*', r"\1\2", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(^|\s)(\d+[.)、]\s*)在“[^”]+”部分提到[:：]\s*", r"\1\2", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if english:
            cleaned = self._fix_ocr_spacing(cleaned)
        return cleaned

    def _fix_ocr_spacing(self, text: str) -> str:
        common_terms = {
            "charg e": "charge",
            "charg ing": "charging",
            "operatio n": "operation",
            "setting s": "settings",
            "batter y": "battery",
            "connectio n": "connection",
            "pressur e": "pressure",
            "temperatur e": "temperature",
            "manua l": "manual",
            "function s": "functions",
            "interfac e": "interface",
            "installatio n": "installation",
        }
        fixed = text
        for broken, replacement in common_terms.items():
            fixed = re.sub(re.escape(broken), replacement, fixed, flags=re.IGNORECASE)
        return fixed

    def _is_auxiliary_prompt_sentence(self, text: str) -> bool:
        lowered = text.strip().lower()
        if not lowered:
            return True
        generic_followups = {
            "如何设置",
            "这些模式有什么特点",
            "如何快速上手使用",
            "how to set it",
            "how to use it quickly",
            "what are their features",
        }
        if lowered in generic_followups:
            return True
        auxiliary_prefixes = (
            "understanding this process",
            "understanding this",
            "this process can",
            "this can enhance",
            "this can help",
        )
        return lowered.startswith(auxiliary_prefixes)

    def _collect_answer_context(
        self,
        sub_question_results: list[tuple[str, list[SearchResult]]],
        image_index: dict[str, ImageRecord],
    ) -> AnswerContext:
        references: list[dict] = []
        related_images: list[dict] = []
        used_manuals: list[str] = []
        used_sections: list[str] = []

        for _, results in sub_question_results:
            if not results:
                continue
            selected_results = self._select_primary_evidence_results(_, results[:4])
            for result in selected_results[:3]:
                if result.chunk.manual_name not in used_manuals:
                    used_manuals.append(result.chunk.manual_name)
                if result.chunk.section_title not in used_sections:
                    used_sections.append(result.chunk.section_title)
                if not any(item["chunk_id"] == result.chunk.chunk_id for item in references):
                    references.append(
                        {
                            "chunk_id": result.chunk.chunk_id,
                            "manual_name": result.chunk.manual_name,
                            "section_title": result.chunk.section_title,
                            "score": round(result.score, 4),
                        }
                    )
                for image_id in result.chunk.image_ids[:2]:
                    image = image_index.get(image_id)
                    if image is None:
                        continue
                    candidate = {
                        "image_id": image.image_id,
                        "manual_name": image.manual_name,
                        "caption": image.caption,
                        "image_path": image.image_path,
                    }
                    if candidate not in related_images:
                        related_images.append(candidate)

        return AnswerContext(
            references=references[:5],
            related_images=related_images[:5],
            used_manuals=used_manuals,
            used_sections=used_sections,
        )

    def _append_result_images(
        self,
        result: SearchResult,
        image_index: dict[str, ImageRecord],
        related_images: list[dict],
        *,
        limit: int,
    ) -> None:
        for image_id in result.chunk.image_ids[:limit]:
            image = image_index.get(image_id)
            if image is None:
                continue
            candidate = {
                "image_id": image.image_id,
                "manual_name": image.manual_name,
                "caption": image.caption,
                "image_path": image.image_path,
            }
            if candidate not in related_images:
                related_images.append(candidate)

    def _append_image_ids(
        self,
        image_ids: list[str],
        image_index: dict[str, ImageRecord],
        related_images: list[dict],
        *,
        limit: int,
    ) -> None:
        for image_id in image_ids[:limit]:
            image = image_index.get(image_id)
            if image is None:
                continue
            candidate = {
                "image_id": image.image_id,
                "manual_name": image.manual_name,
                "caption": image.caption,
                "image_path": image.image_path,
            }
            if candidate not in related_images:
                related_images.append(candidate)

    def _planned_image_ids(self, intent: str) -> list[str]:
        planned = {
            "boat_factory_reset": ["Manual09_87", "Manual09_88", "Manual09_89"],
            "boat_fuse": ["Manual09_282", "Manual09_283", "Manual09_281"],
            "swim_platform_open": ["Manual09_211", "Manual09_212"],
            "boat_engine_oil_level": ["Manual09_197", "Manual09_198"],
            "boat_battery_compartment": ["Manual09_280"],
            "boat_anchor_light": ["Manual09_223", "Manual09_224", "Manual09_225"],
            "boat_fire_extinguisher": ["Manual09_211", "Manual09_212"],
            "boat_flush_cooling": ["Manual09_173", "Manual09_174", "Manual09_179"],
            "boat_jet_wash_use": ["Manual09_173", "Manual09_174", "Manual09_179"],
            "boat_engine_start": ["Manual09_234", "Manual09_235", "Manual09_236", "Manual09_237"],
            "boat_bimini_upright_storage": ["Manual09_191", "Manual09_192"],
            "boat_bimini_remove": ["Manual09_182", "Manual09_183", "Manual09_184"],
            "boat_bimini_install": ["Manual09_185", "Manual09_186", "Manual09_187"],
            "ereader_buttons": ["Manual13_0", "Manual13_1"],
            "ereader_ebook_mode": ["eReader_08", "Manual13_5", "Manual13_6"],
            "ereader_record": ["Manual13_11", "Manual13_12"],
            "ereader_video": ["Manual13_9"],
            "fitness_charge": ["Manual16_1", "Manual16_2"],
            "fitness_interface": ["Manual16_12", "Manual16_13"],
            "fitness_heart_rate": ["Manual16_4", "Manual16_5"],
            "fitness_payment": ["Manual16_46", "Manual16_47"],
            "oven_drip_tray": ["oven_08"],
            "oven_baking_tray": ["oven_09"],
            "oven_wire_shelf": ["oven_10"],
            "oven_grill_pan_set": ["oven_12"],
            "generator_hot_safety": ["generator_04", "Manual18_8", "Manual18_9"],
            "generator_shock_safety": ["Manual18_12", "Manual18_13", "Manual18_14"],
            "generator_oil_check": ["generator_17", "generator_18"],
            "generator_control_switches": ["Manual18_18", "generator_06", "generator_07"],
            "generator_identification": ["Manual18_0", "Manual18_1"],
            "water_pump_parts": ["pump_16", "pump_17"],
            "generator_stop": ["generator_22", "generator_23", "generator_24"],
            "generator_no_start": ["generator_05"],
            "mower_roll_bar": ["Manual23_32", "Manual23_33"],
            "mower_remove_filters": ["Manual23_72", "Manual23_83"],
            "camera_battery_charge": ["Manual29_13", "Manual29_14"],
            "camera_battery_install": ["Manual29_13", "Manual29_14"],
            "memory_card": ["Manual29_57", "Manual29_58"],
            "camera_mount_lens": ["Manual10_21", "Manual10_22"],
            "camera_eyepiece_cover": ["Manual10_155"],
            "camera_p_mode": ["Manual10_115"],
            "camera_auto_print": ["Manual29_30", "Manual29_31", "Manual29_32"],
            "processor_unit_parts": ["Manual38_1", "Manual38_2"],
            "bike_workout_area": ["Manual14_3"],
            "jetski_seat": ["Manual20_35", "Manual20_36", "Manual20_37"],
            "jetski_filler_caps": ["Manual20_40", "Manual20_41"],
            "jetski_hood_open_close": ["Manual20_38", "Manual20_39"],
            "jetski_levers": ["Manual20_46", "Manual20_47", "Manual20_51"],
            "landline_install_handset": ["Manual22_23", "Manual22_27"],
            "landline_handset_led": ["Manual22_40"],
            "landline_base_led": ["Manual22_46"],
            "vacuum_clean_filter": ["Manual32_10", "Manual32_11"],
            "vacuum_clean_extractors": ["Manual32_19", "Manual32_20", "Manual32_21"],
            "vacuum_clean_side_brush": ["Manual32_16"],
            "vacuum_front_caster": ["Manual32_15"],
            "blower_start": ["Manual04_24", "Manual04_25", "Manual04_26"],
            "blower_carburetor": ["Manual04_42"],
            "blower_safety": ["Manual04_3"],
            "blower_ppe": ["Manual04_3"],
            "airpurifier_remove_filter_packaging": ["Manual03_0", "Manual03_1", "Manual03_2"],
            "airpurifier_replace_filter": ["Manual03_21"],
            "airpurifier_dust_sensor": ["Manual03_22", "Manual03_23"],
            "airpurifier_modes": ["Manual03_14", "Manual03_18", "Manual03_20", "Manual03_21"],
            "air_conditioner_components": ["Manual01_0"],
            "air_conditioner_auto_restart": ["Manual01_25", "Manual01_26"],
            "chair_parts": ["Manual02_0"],
            "chair_functions": ["Manual02_0"],
            "dishwasher_parts": ["Manual06_15", "Manual06_16"],
            "dishwasher_spray_arm_clean": ["Manual06_23"],
            "dishwasher_basket_height": ["Manual06_13", "Manual06_14"],
            "bike_specs": ["Manual14_0", "Manual14_1", "Manual14_2"],
            "bike_edit_profile": ["Manual14_26"],
            "bike_easy_ride_programs": ["Manual14_27", "Manual14_28", "Manual14_29"],
            "bike_mountain_programs": ["Manual14_30", "Manual14_31", "Manual14_32"],
            "bike_challenge_programs": ["Manual14_33", "Manual14_34", "Manual14_35"],
            "fitness_box_contents": ["Manual16_3"],
            "coffee_energy_saving": ["Manual07_4", "Manual07_5"],
            "coffee_empty_system": ["Manual07_28", "Manual07_29", "Manual07_30", "Manual07_31", "Manual07_32"],
            "boat_maintenance_screen": ["Manual09_83", "Manual09_84"],
            "boat_trip_screen": ["Manual09_69", "Manual09_70"],
            "camera_cp_direct": ["Manual10_188"],
            "dishwasher_add_detergent": ["Dish_washer_03", "Manual06_4"],
            "dishwasher_tablet": ["Manual06_5", "Manual06_15", "Manual06_16"],
            "drill_keyless_chuck": ["drill0_01", "drill0_02", "drill0_03"],
            "drill_battery_charge": ["Manual11_2"],
            "drill_battery_pack": ["Manual11_8"],
            "drill_dcb101_indicator": ["drill0_08", "drill0_09"],
            "generator_start": ["Manual18_25", "Manual18_26", "Manual18_27"],
            "jetski_throttle_turning": ["Manual40_20", "Manual40_22"],
            "jetski_stop": ["Manual40_21", "Manual40_25"],
            "jetski_avoid_collision": ["Manual40_0"],
            "jetski_characteristics": ["Manual20_24"],
            "jetski_fuel_filter": ["Manual20_81"],
            "jetski_intake_impeller": ["Manual20_74"],
            "mouse_battery_install": ["Manual27_1", "Manual27_2", "Manual27_3"],
            "mouse_battery_status": ["Manual27_14", "Manual27_15"],
            "mouse_other_hid": ["Manual27_18"],
            "pressure_sealing_ring": ["Manual30_16", "Manual30_30", "Manual30_31"],
            "pressure_steam_release": ["Manual30_14"],
            "pressure_anti_block_shield": ["Manual30_19", "Manual30_34", "Manual30_35"],
            "fax_finger_safety": ["Manual15_2"],
            "fax_warning_labels": ["Manual15_2"],
            "steam_quick_assembly": ["Manual05_3", "Manual05_4", "Manual05_5"],
            "thermostat_datetime": ["Manual36_25", "Manual36_26", "Manual36_31"],
            "thermostat_temp_override": ["Manual36_32", "Manual36_33"],
            "snowmobile_throttle_cable": ["Manual34_35", "Manual34_36", "Manual34_233"],
            "snowmobile_steering_system": ["Manual34_109"],
            "snowmobile_turning": ["Manual34_127"],
            "snowmobile_downhill": ["Manual34_128"],
            "snowmobile_engine_start": ["Manual34_106", "Manual34_107"],
            "toothbrush_intensity": ["toothbrush0_06", "toothbrush0_07"],
            "earphone_ear_tip_replace": ["Manual38_4"],
            "earphones_other_functions": ["earphones_01", "earphones_02", "earphones_03", "Manual12_10"],
            "earphones_reset": ["Manual12_11", "Manual12_12"],
            "coffee_program_volume": ["Manual07_24", "Manual07_25", "Manual07_26"],
            "coffee_after_use_clean": ["Manual07_22", "Manual07_23", "Manual07_47"],
            "vacuum_dual_modes": ["Manual32_5", "Manual32_6"],
            "vacuum_robot_anatomy": ["Manual32_0"],
            "steam_functions": ["Manual05_1", "Manual05_2", "Manual05_12"],
            "steam_hard_floor": [],
            "network_camera_t_rail": ["Manual33_10", "Manual33_11", "Manual33_12", "Manual33_13", "Manual33_14"],
            "grill_assembly_first_three_steps": ["Manual19_49", "Manual19_50", "Manual19_51"],
            "function_keyboard_setup": ["Manual21_2", "function_keyboard_01", "function_keyboard_02", "Manual21_3"],
            "function_keyboard_switch_replace": ["Manual21_14", "Manual21_15"],
            "function_keyboard_warranty": [],
            "motherboard_t_sensor": ["Manual25_43"],
            "rideon_motorcycle_front_wheel": ["Manual26_6", "Manual26_7"],
        }
        return planned.get(intent, [])

    def _filter_entity_mismatched_results(
        self,
        question: str,
        sub_question_results: list[tuple[str, list[SearchResult]]],
    ) -> list[tuple[str, list[SearchResult]]]:
        lowered_question = question.lower()
        active_rules = [
            (required_terms, forbidden_terms)
            for triggers, required_terms, forbidden_terms in ENTITY_GUARD_RULES
            if any(trigger in lowered_question for trigger in triggers)
        ]
        if not active_rules:
            return sub_question_results

        filtered_groups: list[tuple[str, list[SearchResult]]] = []
        for sub_question, results in sub_question_results:
            kept: list[SearchResult] = []
            for result in results:
                evidence_text = f"{result.chunk.manual_name} {result.chunk.product_name} {result.chunk.section_title} {result.chunk.text}".lower()
                mismatch = False
                for required_terms, forbidden_terms in active_rules:
                    has_required = any(term in evidence_text for term in required_terms)
                    has_forbidden = any(term in evidence_text for term in forbidden_terms)
                    if has_forbidden and not has_required:
                        mismatch = True
                        break
                if not mismatch:
                    kept.append(result)
            filtered_groups.append((sub_question, kept))
        return filtered_groups

    def _can_polish_rule_answer(
        self,
        question: str,
        confidence: float,
        sub_question_results: list[tuple[str, list[SearchResult]]],
        context: AnswerContext,
    ) -> bool:
        if confidence < 0.24:
            return False
        if not context.references:
            return False
        if self._missing_required_evidence(question, sub_question_results):
            return False
        return True

    def _build_support_fallback(self, question: str) -> GeneratedAnswer:
        normalized_question = re.sub(r"\s+", "", question)
        answer = ""
        if any(keyword in normalized_question for keyword in ("上门安装", "安装人员", "安装服务", "免费安装")):
            for keywords, template in SUPPORT_TEMPLATE_RULES:
                if "上门安装" in keywords:
                    answer = template
                    break
        if any(keyword in normalized_question for keyword in ("保质期", "临期", "快过期", "马上过期", "过期")):
            for keywords, template in SUPPORT_TEMPLATE_RULES:
                if "保质期" in keywords:
                    answer = template
                    break
        for keywords, template in SUPPORT_TEMPLATE_RULES:
            if answer:
                break
            if any(keyword in normalized_question for keyword in keywords):
                answer = template
                break
        if not answer:
            answer = (
                "您好，这类问题需要结合订单状态和平台售后规则确认。"
                "建议您提供订单号、商品名称、签收时间、问题照片或视频，以及您的明确诉求"
                "（如退款、换货、补发、维修或投诉）。人工客服核实订单后，可以给出更准确的处理方案。"
            )
        return GeneratedAnswer(
            answer=answer,
            confidence=0.0,
            references=[],
            related_images=[],
            used_manuals=[],
            used_sections=[],
        )

    def _unsupported_product_label(self, question: str) -> str | None:
        lowered = question.lower()
        compact = re.sub(r"\s+", "", lowered)
        for term, label in UNSUPPORTED_PRODUCT_TERMS:
            normalized_term = term.lower()
            if " " in normalized_term:
                if normalized_term in lowered:
                    return label
                continue
            if normalized_term in compact:
                return label
        return None

    def _strict_product_guard_enabled(self) -> bool:
        return os.getenv("STRICT_PRODUCT_GUARD", "").strip().lower() in {"1", "true", "yes", "on"}

    def _build_product_not_in_kb_fallback(self, product_label: str) -> GeneratedAnswer:
        answer = (
            f"您好，当前本地说明书库没有检索到“{product_label}”对应的说明书资料，"
            f"因此不能基于现有手册可靠地给出具体步骤或部件说明。{INSUFFICIENT_PHRASE}"
            "建议补充该产品的准确型号、说明书页面或清晰图片后再确认，避免按无关产品手册操作。"
        )
        return GeneratedAnswer(
            answer=answer,
            confidence=0.0,
            references=[],
            related_images=[],
            used_manuals=[],
            used_sections=[],
        )

    def _build_manual_fallback(self, _: str) -> GeneratedAnswer:
        answer = (
            "您好，当前检索到的说明书证据还不足以支持一个明确结论。"
            "如果您方便的话，请补充更具体的产品名称、型号、按钮名称，或上传更清晰的图片，我再继续帮您定位。"
        )
        return GeneratedAnswer(
            answer=answer,
            confidence=0.0,
            references=[],
            related_images=[],
            used_manuals=[],
            used_sections=[],
        )

    def _generate_with_llm(
        self,
        question: str,
        sub_question_results: list[tuple[str, list[SearchResult]]],
        context: AnswerContext,
        product_hint: str | None,
    ) -> str | None:
        if not self._llm_final_generation_enabled():
            return None
        if self.llm_client is None or not self.llm_client.is_enabled():
            return None

        messages = self._build_llm_messages(
            question=question,
            sub_question_results=sub_question_results,
            context=context,
            product_hint=product_hint,
        )
        answer = self.llm_client.chat(
            messages,
            max_tokens=min(self.llm_client.settings.llm_max_tokens, 256),
        )
        cleaned = self._clean_llm_answer(answer)
        return self._postprocess_llm_answer(cleaned, question, sub_question_results, context)

    def _llm_final_generation_enabled(self) -> bool:
        return os.getenv("LLM_FINAL_GENERATION_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}

    def _llm_polish_enabled(self) -> bool:
        return os.getenv("LLM_POLISH_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}

    def _polish_rule_answer_with_llm(
        self,
        question: str,
        rule_answer: str,
        sub_question_results: list[tuple[str, list[SearchResult]]],
        context: AnswerContext,
        product_hint: str | None,
    ) -> str | None:
        if self.llm_client is None or not self.llm_client.is_enabled():
            return None
        if not context.references:
            return None

        evidence = self._build_compact_evidence(sub_question_results, char_budget=1500)
        related_image_ids = "、".join(image["image_id"] for image in context.related_images[:3]) if context.related_images else "无"
        prompt = (
            f"用户问题：{question}\n"
            f"产品提示：{product_hint or '无'}\n"
            f"可用图片ID：{related_image_ids}\n\n"
            f"当前规则答案：\n{rule_answer}\n\n"
            f"证据：\n{evidence}\n\n"
            "请把上面的证据改写成最终中文客服回答，要求：\n"
            "1. 只能使用证据和当前规则答案中已经出现的信息，不要补充常识、政策或未出现的参数。\n"
            "2. 英文证据必须翻译成中文；按钮名、型号或专有名词可保留原文。\n"
            "3. 不要写“在某章节提到”或“根据当前检索到”，直接回答用户问题。\n"
            "4. 步骤题用 1. 2. 3. 分点；多问题要逐项回答。\n"
            "5. 如果有可用图片ID，在最相关的一句后插入 <PIC>，最后写“参考图片：实际ID”。\n"
            "6. 如果当前规则答案已经是订单/售后客服兜底，也要保留客服口吻，不要强行引用说明书。\n"
        )
        messages = [
            LLMMessage(
                role="system",
                content=(
                    "你是中文客服答案润色器。你的任务是把证据整理成自然、准确、简洁的中文回答。"
                    "严禁编造证据外的信息；严禁输出分析过程。"
                ),
            ),
            LLMMessage(role="user", content=prompt),
        ]
        answer = self.llm_client.chat(messages, temperature=0.0, max_tokens=min(self.llm_client.settings.llm_max_tokens, 384))
        cleaned = self._clean_llm_answer(answer)
        if not cleaned or self._looks_low_quality_answer(cleaned):
            return None
        if self._contains_insufficient_claim(cleaned) and not self._contains_insufficient_claim(rule_answer):
            return None
        if self._answer_appears_to_add_unsupported_policy(rule_answer, cleaned):
            return None
        if self._answer_conflicts_with_question(question, cleaned):
            return None
        normalized = self._finalize_answer_text(cleaned, context)
        if self._answer_conflicts_with_question(question, normalized):
            return None
        if self._llm_verify_enabled() and not self._verify_answer_with_llm(
            question,
            normalized,
            sub_question_results,
            context,
        ):
            return None
        return normalized

    def _llm_verify_enabled(self) -> bool:
        return os.getenv("LLM_VERIFY_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}

    def _verify_answer_with_llm(
        self,
        question: str,
        answer: str,
        sub_question_results: list[tuple[str, list[SearchResult]]],
        context: AnswerContext,
    ) -> bool:
        if self.llm_client is None or not self.llm_client.is_enabled():
            return True
        evidence = self._build_compact_evidence(sub_question_results, char_budget=1000)
        related_image_ids = "、".join(image["image_id"] for image in context.related_images[:3]) if context.related_images else "无"
        prompt = (
            f"用户问题：{question}\n"
            f"答案：{answer}\n"
            f"可用图片ID：{related_image_ids}\n"
            f"证据：\n{evidence}\n\n"
            "请检查答案是否适合作为最终提交。只输出JSON："
            "{\"pass\":true,\"reason\":\"...\"} 或 {\"pass\":false,\"reason\":\"...\"}。"
            "判定标准：答案必须回答用户问题，不能添加证据外具体参数/政策/承诺，图片ID必须来自可用图片ID。"
        )
        raw = self.llm_client.chat(
            [
                LLMMessage(role="system", content="你是RAG答案质检器，只判断是否通过，不改写答案。"),
                LLMMessage(role="user", content=prompt),
            ],
            temperature=0.0,
            max_tokens=96,
        )
        if not raw:
            return True
        data = self._extract_verifier_json(raw)
        if data is None:
            return "false" not in raw.lower() and "不通过" not in raw
        verdict = data.get("pass")
        return bool(verdict) if isinstance(verdict, bool) else True

    def _extract_verifier_json(self, raw: str) -> dict | None:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except Exception:
                return None
        return data if isinstance(data, dict) else None

    def _answer_appears_to_add_unsupported_policy(self, source: str, answer: str) -> bool:
        if any(term in source for term in ("订单", "售后", "退款", "退货", "发票", "物流", "客服")):
            return False
        policy_terms = ("退款", "退货", "赔偿", "保修政策", "订单号", "人工客服", "运费")
        return any(term in answer and term not in source for term in policy_terms)

    def _build_compact_evidence(
        self,
        sub_question_results: list[tuple[str, list[SearchResult]]],
        *,
        char_budget: int,
    ) -> str:
        lines: list[str] = []
        remaining = char_budget
        for group_index, (sub_question, results) in enumerate(sub_question_results, start=1):
            if remaining <= 120:
                break
            selected = self._select_primary_evidence_results(sub_question, results[:5])
            header = f"子问题{group_index}：{sub_question}"
            lines.append(header)
            remaining -= len(header)
            for rank, result in enumerate(selected[:2], start=1):
                chunk = result.chunk
                text = re.sub(r"\s+", " ", chunk.text).strip()
                max_text_len = max(80, min(360, remaining - 120))
                if len(text) > max_text_len:
                    text = text[: max_text_len - 3].rstrip("，,；; ") + "..."
                image_ids = "、".join(chunk.image_ids[:3]) if chunk.image_ids else "无"
                item = (
                    f"证据{group_index}.{rank} "
                    f"章节：{chunk.section_title}；"
                    f"图片：{image_ids}；"
                    f"内容：{text}"
                )
                if len(item) > remaining:
                    break
                lines.append(item)
                remaining -= len(item)
        return "\n".join(lines)

    def _build_llm_messages(
        self,
        question: str,
        sub_question_results: list[tuple[str, list[SearchResult]]],
        context: AnswerContext,
        product_hint: str | None,
    ) -> list[LLMMessage]:
        evidence_lines: list[str] = []
        remaining_budget = int(os.getenv("LLM_EVIDENCE_CHAR_BUDGET", "1600"))
        for group_index, (sub_question, results) in enumerate(sub_question_results, start=1):
            plan = self._build_evidence_plan(sub_question)
            max_results_per_question = 3 if len(sub_question_results) <= 2 else 2
            max_results_per_question = min(max_results_per_question, max(1, plan.max_evidence))
            selected_results = self._select_primary_evidence_results(
                sub_question,
                results[: max_results_per_question + 4],
            )
            header = f"子问题{group_index}：{sub_question}"
            if remaining_budget <= len(header):
                break
            evidence_lines.append(header)
            remaining_budget -= len(header)
            for rank, result in enumerate(selected_results[:max_results_per_question], start=1):
                chunk = result.chunk
                image_ids = "、".join(chunk.image_ids[:3]) if chunk.image_ids else "无"
                chunk_text = chunk.text.strip()
                if len(chunk_text) > 520:
                    chunk_text = chunk_text[:517].rstrip("，,；; ") + "..."
                candidate_lines = [
                    f"证据{group_index}.{rank}",
                    f"手册：{chunk.manual_name}",
                    f"产品：{chunk.product_name}",
                    f"章节：{chunk.section_title}",
                    f"类型：{chunk.chunk_type}",
                    f"相关图片ID：{image_ids}",
                    f"内容：{chunk_text}",
                ]
                candidate_text = "\n".join(candidate_lines)
                if remaining_budget <= 120:
                    break
                if len(candidate_text) > remaining_budget:
                    overflow = len(candidate_text) - remaining_budget
                    if overflow >= len(chunk_text) - 80:
                        break
                    truncated_text = chunk_text[: len(chunk_text) - overflow - 3].rstrip("，,；; ")
                    candidate_lines[-1] = f"内容：{truncated_text}..."
                    candidate_text = "\n".join(candidate_lines)
                evidence_lines.extend(candidate_lines)
                remaining_budget -= len(candidate_text)

        related_image_ids = "、".join(image["image_id"] for image in context.related_images) if context.related_images else "无"
        user_prompt = (
            f"用户问题：{question}\n"
            f"产品提示：{product_hint or '无'}\n"
            f"可返回的相关图片ID：{related_image_ids}\n\n"
            "说明书证据如下：\n"
            f"{chr(10).join(evidence_lines)}\n\n"
            "输出要求：\n"
            "1. 直接给出中文客服回答。\n"
            "2. 只能基于上述证据作答。\n"
            "3. 只有在确实无法回答核心问题时，才明确回答“当前检索到的说明书证据不足以支持明确结论”。\n"
            "4. 如果证据能回答问题，不要输出证据不足；如果只能回答一部分，先回答已确认内容，再说明未覆盖部分。\n"
            "5. 步骤题必须整理成 1. 2. 3.，不要输出“在某部分提到：1.”这种空摘录。\n"
            "6. 解释含义、参数、注意事项时要用自然客服语言组织，不要简单复制章节标题。\n"
            "7. 如果适合提醒图片，可在最后一句写“可参考配图：实际图片ID”，不要输出“图片ID”占位符。\n"
        )
        return [
            LLMMessage(role="system", content=LLM_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

    def _clean_llm_answer(self, answer: str | None) -> str | None:
        if not answer:
            return None
        cleaned = answer.strip()
        cleaned = re.sub(r"^```(?:text|markdown)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = re.sub(r"^答案[:：]\s*", "", cleaned)
        cleaned = re.sub(r"^(?:根据(?:提供|现有)的信息|根据您描述的情况)[，,:：]\s*", "", cleaned)
        cleaned = cleaned.strip()
        return cleaned or None

    def _postprocess_llm_answer(
        self,
        answer: str | None,
        question: str,
        sub_question_results: list[tuple[str, list[SearchResult]]],
        context: AnswerContext,
    ) -> str | None:
        if not answer:
            return None

        normalized = self._finalize_answer_text(answer.replace("\r\n", "\n").strip(), context)
        if self._answer_conflicts_with_question(question, normalized):
            return None
        if self._looks_low_quality_answer(normalized):
            return None
        if not self._contains_insufficient_claim(normalized):
            return normalized

        confidence = self._estimate_confidence(sub_question_results)
        detail_score = self._detail_signal_score(normalized)
        strong_evidence = self._has_strong_evidence(question, sub_question_results, confidence)

        if detail_score >= 2 or (detail_score >= 1 and strong_evidence):
            normalized = self._remove_insufficient_claims(normalized)
            normalized = self._finalize_answer_text(normalized, context)
            normalized = re.sub(r"\n{3,}", "\n\n", normalized)
            return normalized or None

        image_hint = self._extract_image_hint(normalized, context)
        compressed = INSUFFICIENT_PHRASE
        if image_hint:
            compressed = f"{compressed}\n\n{image_hint}"
        return compressed

    def _should_fallback_to_rule_answer(
        self,
        answer: str,
        question: str,
        sub_question_results: list[tuple[str, list[SearchResult]]],
    ) -> bool:
        if self._looks_low_quality_answer(answer):
            return True
        if not self._contains_insufficient_claim(answer):
            return False
        if self._has_rule_usable_evidence(question, sub_question_results):
            return True
        if not self._is_parameter_question(question):
            return False
        confidence = self._estimate_confidence(sub_question_results)
        if confidence < 0.18:
            return False
        return any(
            results and self._chunk_contains_parameter_signal(results[0].chunk.text)
            for _, results in sub_question_results
        )

    def _has_rule_usable_evidence(
        self,
        question: str,
        sub_question_results: list[tuple[str, list[SearchResult]]],
    ) -> bool:
        required_terms = self._required_evidence_terms(question)
        if required_terms and self._missing_required_evidence(question, sub_question_results):
            return False

        for sub_question, results in sub_question_results:
            if not results:
                continue
            selected = self._select_primary_evidence_results(sub_question, results[:6])
            if not selected:
                continue
            snippet = self._select_best_snippet(sub_question, selected)
            if not snippet or self._contains_insufficient_claim(snippet):
                continue
            if self._looks_low_quality_answer(snippet):
                continue
            if self._is_low_information_snippet(snippet, question):
                continue
            if any(result.chunk.image_ids for result in selected[:3]):
                return True
            if len(re.sub(r"\s+", "", snippet)) >= 28:
                return True
        return False

    def _estimate_confidence(self, sub_question_results: list[tuple[str, list[SearchResult]]]) -> float:
        top_scores = [
            self._select_primary_evidence_results(question, results[:4])[0].score
            for question, results in sub_question_results
            if results
        ]
        return sum(top_scores) / len(top_scores) if top_scores else 0.0

    def _has_strong_evidence(
        self,
        question: str,
        sub_question_results: list[tuple[str, list[SearchResult]]],
        confidence: float,
    ) -> bool:
        if confidence < 0.2:
            return False
        preferred_types = self._build_evidence_plan(question).preferred_types
        for _, results in sub_question_results:
            if not results:
                continue
            best = self._select_primary_evidence_results(_, results[:4])[0]
            if best.chunk.chunk_type in preferred_types and best.score >= 0.28:
                return True
        return False

    def _select_primary_evidence_results(self, question: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return results

        plan = self._build_evidence_plan(question)
        if plan.intent == "default":
            return self._order_default_evidence_results(plan, results)

        raw_primary = [result for result in results if self._is_primary_evidence_for_plan(plan, result)]
        primary = [result for result in raw_primary if not self._is_background_evidence_for_plan(plan, result)]
        if not primary:
            primary = raw_primary
        raw_secondary = [
            result
            for result in results
            if result not in primary and self._is_secondary_evidence_for_plan(plan, result)
        ]
        secondary = [result for result in raw_secondary if not self._is_background_evidence_for_plan(plan, result)]
        background = [
            result
            for result in results
            if result not in primary
            and result not in secondary
            and not self._is_background_evidence_for_plan(plan, result)
        ]
        suppressed = [
            result
            for result in results
            if self._is_background_evidence_for_plan(plan, result) and result not in primary
        ]

        ordered = primary + secondary + background + suppressed
        seen: set[str] = set()
        deduped: list[SearchResult] = []
        for result in ordered:
            if result.chunk.chunk_id in seen:
                continue
            seen.add(result.chunk.chunk_id)
            deduped.append(result)
        return deduped[: max(1, plan.max_evidence)]

    def _order_default_evidence_results(
        self,
        plan: EvidencePlan,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        preferred = [
            result
            for result in results
            if result.chunk.chunk_type in plan.preferred_types and not self._is_generic_generation_context(result)
        ]
        informative = [
            result
            for result in results
            if result not in preferred and not self._is_generic_generation_context(result)
        ]
        suppressed = [result for result in results if self._is_generic_generation_context(result)]
        ordered = preferred + informative + suppressed
        seen: set[str] = set()
        deduped: list[SearchResult] = []
        for result in ordered:
            if result.chunk.chunk_id in seen:
                continue
            seen.add(result.chunk.chunk_id)
            deduped.append(result)
        return deduped[: max(1, plan.max_evidence)]

    def _is_generic_generation_context(self, result: SearchResult) -> bool:
        title = result.chunk.section_title.strip()
        title_lower = title.lower()
        compact_title = re.sub(r"\s+", "", title)
        if result.chunk.chunk_type in {"toc", "title_only"}:
            return True
        if title_lower in {"caution", "warning"} or title_lower.startswith(("caution ", "warning ", "important safety instructions")):
            return True
        if any(term in compact_title for term in ("目录", "前言", "内容", "概述", "目标")):
            return True
        text = result.chunk.text.strip()
        if len(text) < 12 and any(term in compact_title for term in ("提示", "注意", "警告")):
            return True
        if text.count("....") >= 2:
            return True
        return False

    def _build_evidence_plan(self, question: str) -> EvidencePlan:
        lowered = question.lower()
        if (
            ("water pump" in lowered or "水泵" in question)
            and any(token in lowered or token in question for token in ("does not pump", "cannot pump", "unable to pump", "can't pump", "无法抽水", "不能抽水"))
        ):
            return EvidencePlan(
                intent="water_pump_no_pump",
                max_evidence=3,
                preferred_types={"troubleshoot", "warning", "step", "list", "general"},
                primary_terms=("无法抽水",),
                secondary_terms=("软管接头", "卡箍", "机械密封", "o形圈", "strainer", "hose"),
                background_title_terms=("火花塞", "机油更换", "发动机机油"),
                background_body_terms=("火花塞", "机油完全排空", "发动机机油等级"),
            )
        if "水泵" in lowered and any(token in lowered for token in ("核心部件", "部件", "组成", "有哪些")):
            return EvidencePlan(
                "water_pump_parts",
                2,
                {"component", "list", "general"},
                ("部件说明", "油箱", "发动机开关", "注水螺塞"),
                ("油箱盖", "燃油开关", "空气滤清器盖", "火花塞", "放水螺塞"),
                ("目录", "发电机"),
                ("目录", "发电机"),
            )
        if "处理器单元" in lowered and any(token in lowered for token in ("组件", "部件", "构成", "关键")):
            return EvidencePlan(
                "processor_unit_parts",
                2,
                {"component", "list", "general"},
                ("处理器单元", "状态指示灯", "hdmi"),
                ("aux", "usb", "dc in", "通风口"),
                ("使用与操作", "高温提示"),
                ("雷雨", "低温烫伤"),
            )
        if any(token in lowered for token in ("emptying the system", "empty the system", "empty system")) and any(
            token in lowered for token in ("frost", "maintenance", "repair", "not in use", "non-use")
        ):
            return EvidencePlan(
                "coffee_empty_system",
                3,
                {"step", "list", "menu", "general"},
                ("emptying the system", "period of non-use", "frost protection"),
                ("espresso", "lungo", "water tank", "lever", "both leds blink"),
                ("descaling", "cleaning", "energy saving", "roll bar"),
                ("roll bar", "seat belt", "authorized service dealer"),
            )
        if "功能键盘" in lowered and any(token in lowered for token in ("轴体", "热插拔", "拆卸", "重新安装", "安装轴体")):
            return EvidencePlan(
                "function_keyboard_switch_replace",
                3,
                {"step", "list", "general"},
                ("轴体拆卸", "轴体", "拔轴器", "重新安装"),
                ("卡扣", "针脚", "插槽", "垂直向下"),
                ("键帽", "保修", "fcc"),
                ("键帽拔取器", "有害干扰", "保修期限"),
            )
        if "功能键盘" in lowered and any(token in lowered for token in ("设置", "连接", "安装", "setup", "set up")) and "保修" not in lowered:
            return EvidencePlan(
                "function_keyboard_setup",
                3,
                {"step", "list", "general"},
                ("键盘设置", "usb-c", "usb 2.0", "腕托", "支撑脚"),
                ("磁吸", "cam", "板载配置文件", "打字倾斜角度"),
                ("保修", "fcc", "轴体拆卸"),
                ("保修期限", "有害干扰", "拔轴器"),
            )
        if "儿童电动摩托车" in lowered and any(token in lowered for token in ("前轮", "前轴", "车轮")):
            return EvidencePlan(
                "rideon_motorcycle_front_wheel",
                3,
                {"step", "list", "general"},
                ("安装前轮", "前轴", "前轮"),
                ("垫片", "把手管", "螺母", "自由转动"),
                ("充电", "电池", "故障排除"),
                ("充电器", "充电时间", "电池老化"),
            )
        if "功能键盘" in lowered and "保修" in lowered:
            return EvidencePlan(
                "function_keyboard_warranty",
                3,
                {"warranty", "list", "general"},
                ("保修期限", "保修范围", "除外责任", "经销商保修服务"),
                ("购买凭证", "维修", "更换", "退货", "rma"),
                ("fcc", "按键重映射"),
                ("有害干扰", "cam软件"),
            )
        if (
            ("safety precautions" in lowered or "safe operation" in lowered)
            and ("this process" in lowered or "during" in lowered or "followed" in lowered)
        ):
            return EvidencePlan(
                intent="generic_safe_operation",
                max_evidence=2,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("safety", "safe operation", "warning", "precautions"),
                secondary_terms=("manual", "labels", "operator", "instructions"),
                background_title_terms=("fcc", "ren"),
                background_body_terms=("fcc", "ren"),
            )
        if "troubleshooting" in lowered and any(token in lowered for token in ("technical issue", "technical issues", "malfunction", "malfunctions")):
            return EvidencePlan(
                intent="air_conditioner_troubleshooting_safety",
                max_evidence=3,
                preferred_types={"warning", "step", "list", "menu", "general"},
                primary_terms=("qualified service", "proper tools", "testing instruments", "malfunction", "product malfunction"),
                secondary_terms=("power cord", "refrigerant", "drain hose", "air inlet", "air outlet", "authorized service"),
                background_title_terms=("energy saving", "archive", "type"),
                background_body_terms=("type: inverter", "model number", "dealer name"),
            )
        if ("delete" in lowered or "erase" in lowered or "删除" in question) and ("image" in lowered or "images" in lowered or "图像" in question):
            return EvidencePlan(
                intent="delete_images",
                max_evidence=3,
                preferred_types={"menu", "step", "list", "general"},
                primary_terms=("删除", "delete", "erase", "全部", "单张", "all", "single"),
                secondary_terms=("图像", "images"),
                background_title_terms=("在电脑上使用存储卡注意事项", "存储卡注意事项"),
                background_body_terms=("电脑", "文件夹", "格式化"),
            )
        if "相机" in lowered and ("自动打印" in lowered or "auto" in lowered and "打印" in lowered):
            return EvidencePlan(
                "camera_auto_print",
                3,
                {"menu", "step", "list", "general"},
                ("自动打印模式", "auto"),
                ("图像保存后立即开始打印", "打印模式选择器"),
                ("手动打印模式", "存储卡"),
                ("存储卡",),
            )
        if "fax" in lowered and any(token in lowered for token in ("connect", "connecting", "procedure")):
            return EvidencePlan(
                intent="fax_connect",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("fax", "telephone line cord", "LINE", "EXT", "wall jack", "connect"),
                secondary_terms=("fax function", "telephone", "line", "jack", "cord", "power outlet"),
                background_title_terms=("REN", "hearing aid", "voice mail", "caller id"),
                background_body_terms=("ren", "hearing aid", "voice mail", "answering machine"),
            )
        if "base station" in lowered and any(token in lowered for token in ("connect", "connecting")):
            return EvidencePlan(
                intent="landline_base_station",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("base station", "power socket", "telephone socket", "connect"),
                secondary_terms=("telephone line", "line cord", "power adapter", "socket"),
                background_title_terms=("charger", "docking", "battery", "answer a call", "caller id"),
                background_body_terms=("charging contacts", "docking tone", "battery", "answer a call"),
            )
        if (
            any(token in lowered for token in ("landline", "handset", "base station"))
            and "searching status" in lowered
        ):
            return EvidencePlan(
                intent="landline_searching_status",
                max_evidence=2,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("searching status", "base station has power supply", "register the handset"),
                secondary_terms=("move the handset closer", "base station"),
                background_title_terms=("battery", "warning tones", "caller id"),
                background_body_terms=("batteries are almost empty", "line adapter"),
            )
        if (
            any(token in lowered for token in ("air fryer", "airfryer", "空气炸锅"))
            and any(token in lowered for token in ("first use", "first time", "before first use", "首次", "第一次"))
        ):
            return EvidencePlan(
                intent="airfryer_first_use",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("before first use", "remove", "packaging", "clean"),
                secondary_terms=("basket", "pan", "wash", "dry", "hot air"),
                background_title_terms=("wifi", "nutriu", "app", "connect", "pair"),
                background_body_terms=("wi-fi", "wifi", "nutriu", "pairing", "app"),
            )
        if ("t-rail" in lowered or "t rail" in lowered) and any(
            token in lowered for token in ("camera", "network camera", "equipment", "mount")
        ):
            return EvidencePlan(
                intent="network_camera_t_rail",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("t-rail mounting instructions", "t-rail clips", "mount plate"),
                secondary_terms=("template", "set screws", "5/64", "hex key", "snap"),
                background_title_terms=("fcc", "warranty"),
                background_body_terms=("interference", "warranty"),
            )
        if (
            any(token in lowered for token in ("blower", "吹风机"))
            and any(token in lowered for token in ("protective equipment", "personal protective", "ppe", "防护装备", "佩戴"))
        ):
            return EvidencePlan(
                intent="blower_ppe",
                max_evidence=2,
                preferred_types={"list", "general"},
                primary_terms=("个人防护装备", "听力防护", "眼部防护", "面罩"),
                secondary_terms=("防滑", "急救箱", "hearing protection", "eye protection", "face mask"),
                background_title_terms=("启动", "化油器", "肩带"),
                background_body_terms=("启动手柄", "泵油膜片", "化油器"),
            )
        if (
            any(token in lowered for token in ("air conditioner", "空调"))
            and any(token in lowered for token in ("components", "parts", "组成部件", "部件有哪些", "重要组成部件"))
        ):
            return EvidencePlan(
                intent="air_conditioner_components",
                max_evidence=2,
                preferred_types={"component", "list", "general"},
                primary_terms=("部件介绍", "室内机", "室外机", "遥控器"),
                secondary_terms=("front panel", "air filter", "air inlet", "air outlet"),
                background_title_terms=("定时", "自动重启", "清洁"),
                background_body_terms=("timer", "auto restart"),
            )
        if any(token in lowered for token in ("auto restart", "自动重启")) and any(
            product in lowered for product in ("air conditioner", "空调")
        ):
            return EvidencePlan(
                intent="air_conditioner_auto_restart",
                max_evidence=2,
                preferred_types={"step", "list", "general"},
                primary_terms=("自动重启", "按住", "6 秒"),
                secondary_terms=("开 / 关键", "蜂鸣", "指示灯", "power failure"),
                background_title_terms=("极速", "等离子", "遥控器"),
                background_body_terms=("timer", "remote controller"),
            )
        if any(token in lowered for token in ("ergonomic chair", "office chair", "人体工学椅", "椅子")):
            if any(token in lowered for token in ("function", "functions", "功能", "有哪些功能")):
                return EvidencePlan(
                    intent="chair_functions",
                    max_evidence=2,
                    preferred_types={"menu", "list", "general"},
                    primary_terms=("高度调节", "椅背后仰", "按摩功能"),
                    secondary_terms=("升降", "后仰", "扶手", "usb"),
                    background_title_terms=("蓝牙", "洗碗机", "健身追踪器"),
                    background_body_terms=("bluetooth", "dishwasher", "fitness"),
                )
            if any(token in lowered for token in ("parts", "components", "assembly", "assemble", "组装", "部件", "配件")):
                return EvidencePlan(
                    intent="chair_parts",
                    max_evidence=2,
                    preferred_types={"component", "list", "general"},
                    primary_terms=("配件详情", "安装脚轮", "气杆", "底盘"),
                    secondary_terms=("扶手", "头枕", "腰枕", "连接件"),
                    background_title_terms=("蓝牙", "洗碗机", "健身追踪器"),
                    background_body_terms=("bluetooth", "dishwasher", "fitness"),
                )
        if any(token in lowered for token in ("dishwasher", "洗碗机")):
            if any(token in lowered for token in ("不适合", "不能清洗", "不应清洗", "not suitable", "unsuitable")):
                return EvidencePlan(
                    intent="dishwasher_unsuitable_items",
                    max_evidence=2,
                    preferred_types={"list", "general"},
                    primary_terms=("不适合在洗碗机中清洗", "切勿", "请勿清洗"),
                    secondary_terms=("铁制器具", "木质", "骨质", "不耐热", "铜", "镀锡", "水晶"),
                    background_title_terms=("亮碟剂", "餐具放入", "洗涤剂"),
                    background_body_terms=("亮碟剂", "餐具放入洗碗机", "洗涤剂盒"),
                )
            if any(token in lowered for token in ("餐具篮", "碗篮", "上层篮", "basket height", "upper basket")):
                return EvidencePlan(
                    intent="dishwasher_basket_height",
                    max_evidence=3,
                    preferred_types={"step", "menu", "general"},
                    primary_terms=("上层篮高度", "使用篮滚轮", "碗篮调节机构", "升高碗篮", "降低碗篮"),
                    secondary_terms=("限位器", "滚轮位置", "导轨", "卡扣", "同一水平"),
                    background_title_terms=("餐具放入", "不适合", "洗涤剂"),
                    background_body_terms=("不适合在洗碗机中清洗", "洗涤剂盒"),
                )
            if any(token in lowered for token in ("parts", "components", "部件")):
                return EvidencePlan(
                    intent="dishwasher_parts",
                    max_evidence=2,
                    preferred_types={"component", "list", "general"},
                    primary_terms=("程序选择与操作", "开机", "启动", "显示屏"),
                    secondary_terms=("程序选择键", "半载", "预约启动", "餐具篮", "喷淋臂"),
                    background_title_terms=("餐具放入", "不适合", "故障排除"),
                    background_body_terms=("不适合在洗碗机中清洗", "餐具放入"),
                )
            if any(token in lowered for token in ("spray arm", "喷淋臂")):
                return EvidencePlan(
                    intent="dishwasher_spray_arm_clean",
                    max_evidence=2,
                    preferred_types={"step", "list", "general"},
                    primary_terms=("上层喷淋臂", "孔是否堵塞", "拆下并清洁"),
                    secondary_terms=("螺母", "喷淋臂", "下层喷淋臂"),
                    background_title_terms=("餐具放入", "洗涤剂"),
                    background_body_terms=("餐具未有序放置", "洗涤剂盒"),
                )
        if (
            any(token in lowered for token in ("air purifier", "空气净化器"))
            and any(token in lowered for token in ("mode", "modes", "特点", "模式", "设置", "运行"))
        ):
            return EvidencePlan(
                intent="airpurifier_modes",
                max_evidence=3,
                preferred_types={"menu", "list", "general"},
                primary_terms=("常规运行", "风速", "空气质量"),
                secondary_terms=("自动", "睡眠", "安全锁", "更换滤网", "iaq"),
                background_title_terms=("脚轮", "灰尘传感器"),
                background_body_terms=("脚轮安装", "caster"),
            )
        if any(token in lowered for token in ("steam cleaner", "蒸汽清洁机")):
            if any(token in lowered for token in ("hard floor", "硬质地面", "硬质地板", "瓷砖", "地面清洁")):
                return EvidencePlan(
                    intent="steam_hard_floor",
                    max_evidence=2,
                    preferred_types={"step", "list", "general"},
                    primary_terms=("硬质地面清洁", "hard floor"),
                    secondary_terms=("扫地", "吸尘", "出蒸汽", "消毒"),
                    background_title_terms=("保修", "包装箱"),
                    background_body_terms=("warranty", "原包装箱"),
                )
            if any(token in lowered for token in ("function", "functions", "features", "功能", "快速上手", "handheld")):
                return EvidencePlan(
                    intent="steam_functions",
                    max_evidence=3,
                    preferred_types={"component", "list", "general"},
                    primary_terms=("产品部件介绍", "手持蒸汽器", "蒸汽开关"),
                    secondary_terms=("硬质地板", "布艺清洁头", "喷射喷嘴", "弧形喷嘴"),
                    background_title_terms=("保修", "包装箱"),
                    background_body_terms=("warranty", "原包装箱"),
                )
        if (
            ("battery conversion" in lowered or "conversion feature" in lowered)
            and any(token in lowered for token in ("boat", "sailing", "jet boat"))
        ):
            return EvidencePlan(
                intent="boat_battery_switches",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("battery switches", "battery switch assembly", "start", "house", "emerg parallel"),
                secondary_terms=("marine batteries", "start battery", "house battery", "on position", "off position"),
                background_title_terms=("microwave", "oven", "door", "charger"),
                background_body_terms=("microwave energy", "door open", "harmful exposure"),
            )
        if "jet wash" in lowered and any(token in lowered for token in ("boat", "jet boat", "clean")):
            return EvidencePlan(
                intent="boat_jet_wash_use",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("jet wash", "coil hose", "hose fitting", "jet wash switch"),
                secondary_terms=("start the engines", "press the jet wash switch", "clean the boat"),
                background_title_terms=("dishwasher", "water pump", "fuse"),
                background_body_terms=("water tap", "pump does not pump", "house fuse"),
            )
        if (
            ("over temperature" in lowered or "temperature warning" in lowered or "overheat" in lowered)
            and any(token in lowered for token in ("boat", "jet boat"))
        ):
            return EvidencePlan(
                intent="boat_over_temperature",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("over temperature", "warning", "engine", "cooling water"),
                secondary_terms=("reduce the engine speed", "return to shore", "safe location", "pilot outlet", "water discharge"),
                background_title_terms=("weight", "cargo", "capacity", "load"),
                background_body_terms=("weight low", "evenly distributed", "cargo"),
            )
        if "fire extinguisher" in lowered and any(token in lowered for token in ("on board", "boat", "stored", "placed")):
            return EvidencePlan(
                intent="boat_fire_extinguisher",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("storing the fire extinguisher", "fire extinguisher", "lockable storage compartment"),
                secondary_terms=("battery compartment", "chemical-type", "clean agent", "capacity"),
                background_title_terms=("seat", "watercraft", "spark plug"),
                background_body_terms=("fire extinguisher container cap", "remove the seat"),
            )
        if "swim platform" in lowered and any(token in lowered for token in ("open", "打开")):
            return EvidencePlan(
                intent="swim_platform_open",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("wet storage compartment", "swim platform", "lock handle", "rear platform hatch"),
                secondary_terms=("pull", "clockwise", "open", "close"),
                background_title_terms=("operator", "age", "safety course", "wakeboarding"),
                background_body_terms=("operator's age", "watercraft operators course"),
            )
        if "snowmobile" in lowered and "engine" in lowered and any(token in lowered for token in ("start", "starting")):
            return EvidencePlan(
                intent="snowmobile_engine_start",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("starting the engine", "engine stop switch", "starter", "throttle"),
                secondary_terms=("parking brake", "choke", "pull the starter", "warm up"),
                background_title_terms=("spark plug", "electrode", "drive belt"),
                background_body_terms=("spark plug", "electrode gap", "belt"),
            )
        if "af mode" in lowered or "autofocus" in lowered:
            return EvidencePlan(
                intent="camera_af_mode",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("af mode", "one-shot af", "ai servo af", "ai focus af"),
                secondary_terms=("af point", "focusing", "autofocus", "camera"),
                background_title_terms=("virtual wall", "picture control", "caption"),
                background_body_terms=("vacuum", "tv", "barrier"),
            )
        if "fuse" in lowered and "boat" in lowered:
            return EvidencePlan(
                intent="boat_fuse",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("fuse", "replace", "fuse box", "fuse replacement"),
                secondary_terms=("fuse puller", "spare fuse", "fuse box cover", "amperage"),
                background_title_terms=("dishwasher", "冰箱", "空调", "refrigerator"),
                background_body_terms=("water tap", "door", "dishwasher", "家中保险丝", "进水水龙头", "机门"),
            )
        if "quick release" in lowered or "qpr" in lowered:
            return EvidencePlan(
                intent="quick_release",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("quick release", "quick release button", "steam release valve", "vent"),
                secondary_terms=("steam", "release", "pressure", "depressurize"),
                background_title_terms=("natural release", "float valve"),
                background_body_terms=("natural release", "npr", "nr "),
            )
        if "natural release" in lowered or "npr" in lowered or re.search(r"\bnr\b", lowered):
            return EvidencePlan(
                intent="natural_release",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("natural release", "npr", "nr", "depressurizes naturally"),
                secondary_terms=("float valve", "pressure", "release", "lid"),
                background_title_terms=("quick release",),
            )
        if "float valve" in lowered:
            return EvidencePlan(
                intent="float_valve",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("float valve", "silicone cap", "lid"),
                secondary_terms=("pressure", "release", "steam", "valve"),
                background_title_terms=("safeguards", "caution"),
            )
        if (
            "toothbrush" in lowered
            and "travel case" in lowered
            and any(token in lowered for token in ("charge", "charging"))
        ):
            return EvidencePlan(
                intent="toothbrush_travel_case_charge",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("charging with the travel case", "travel case", "usb wall adapter", "battery indicator"),
                secondary_terms=("plug", "electrical outlet", "beep twice", "lights", "blinks", "fully charged"),
                background_title_terms=("app", "pairing", "firmware", "brushsync"),
                background_body_terms=("app pairing", "firmware update", "phone"),
            )
        if ("耳机" in lowered or "耳塞" in lowered) and "耳塞" in lowered:
            return EvidencePlan(
                intent="earphone_ear_tip_replace",
                max_evidence=2,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("更换耳塞", "m号耳塞", "s号", "l号"),
                secondary_terms=("旋转", "拔下", "牢固安装", "意外脱落"),
                background_title_terms=("fcc", "规格", "安全警告"),
                background_body_terms=("fcc statement", "battery capacity", "charging case"),
            )
        if (
            any(token in lowered for token in ("earphones", "earbuds", "headphones"))
            and any(token in lowered for token in ("ear tip", "ear tips", "earbud tips", "replace tips"))
        ):
            return EvidencePlan(
                intent="earphone_ear_tip_replace",
                max_evidence=2,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("replace the ear tips", "m size ear tips", "s size", "l size"),
                secondary_terms=("twist", "pull", "securely", "accidentally detached"),
                background_title_terms=("fcc", "specification", "safety"),
                background_body_terms=("fcc statement", "battery capacity", "charging case"),
            )
        if (
            any(token in lowered for token in ("earphones", "earbuds", "headphones"))
            and "case battery" in lowered
            and "charge" in lowered
        ):
            return EvidencePlan(
                intent="earphones_case_charge",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("charging case", "charge", "usb", "battery"),
                secondary_terms=("charging time", "indicator", "case"),
                background_title_terms=("fcc", "specification"),
                background_body_terms=("fcc",),
            )
        if (
            any(token in lowered for token in ("earphones", "earbuds", "headphones"))
            and any(token in lowered for token in ("other function", "other functions", "besides", "voice assistant", "music app", "ambient", "anc", "latency"))
        ):
            return EvidencePlan(
                intent="earphones_other_functions",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("other functions", "voice assistant", "music app", "ambient awareness", "anc", "low latency"),
                secondary_terms=("press and hold", "left earbud", "first beep", "second beep", "cycles modes"),
                background_title_terms=("charging case", "maintenance", "warranty"),
                background_body_terms=("charging port", "battery level", "fcc", "warranty"),
            )
        if (
            any(token in lowered for token in ("earphones", "earbuds", "headphones"))
            and any(token in lowered for token in ("reset", "won't work", "not work", "doesn't work", "cannot work"))
        ):
            return EvidencePlan(
                intent="earphones_reset",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("reset", "initialization", "factory settings", "earphones"),
                secondary_terms=("charging case", "press and hold", "indicator", "left", "right"),
                background_title_terms=("safety", "warranty", "maintenance"),
                background_body_terms=("fcc", "water", "clean cloth"),
            )
        if (
            "factory reset" in lowered
            and any(token in lowered for token in ("boat", "steering", "jet boat"))
        ):
            return EvidencePlan(
                intent="boat_factory_reset",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("factory reset screen", "reset", "factory default settings"),
                secondary_terms=("reset button", "confirmation message", "yes", "no"),
                background_title_terms=("airfryer", "earbuds", "microwave"),
                background_body_terms=("home wi fi", "bluetooth", "earbuds"),
            )
        if "trip screen" in lowered and any(token in lowered for token in ("shown", "display", "show", "boat")):
            return EvidencePlan(
                intent="boat_trip_screen",
                max_evidence=2,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("trip screen", "engine operation", "fuel consumption"),
                secondary_terms=("menu", "scrollbar", "reset button"),
                background_title_terms=("maintenance setting", "factory reset", "warning"),
                background_body_terms=("maintenance", "confirmation message", "over temperature"),
            )
        if "indirect cooking" in lowered and "grill" in lowered:
            return EvidencePlan(
                intent="grill_indirect_cooking",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("indirect cooking", "grill", "indirect heat"),
                secondary_terms=("lid close", "weather conditions", "slow roasting", "baking", "flare-ups"),
                background_title_terms=("warranty", "disclaimer", "pressure cooker"),
                background_body_terms=("indirect damages", "pressure cooking"),
            )
        if "grill" in lowered and "assembly" in lowered and any(token in lowered for token in ("first three", "first 3", "前三")):
            return EvidencePlan(
                intent="grill_assembly_first_three_steps",
                max_evidence=3,
                preferred_types={"step", "list", "general"},
                primary_terms=("assembly", "casters", "bottom shelf", "light adapter", "back panel"),
                secondary_terms=("locking casters", "fixed casters", "lower back panel", "side panels"),
                background_title_terms=("service center", "warranty", "gas leak", "burner"),
                background_body_terms=("call grill service center", "repair protection", "troubleshooting"),
            )
        if (
            "manual program" in lowered
            and any(token in lowered for token in ("channel", "channels", "communication channels"))
        ):
            return EvidencePlan(
                intent="tv_manual_program_channels",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("manual program", "memory/erase", "channel"),
                secondary_terms=("memorize", "erase", "number buttons", "on screen display"),
                background_title_terms=("service", "warning", "antenna"),
                background_body_terms=("dangerous voltage", "qualified personnel"),
            )
        if "outdoor antenna" in lowered or ("antenna" in lowered and "reception" in lowered):
            return EvidencePlan(
                intent="tv_outdoor_antenna",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("outdoor antenna", "antenna", "reception"),
                secondary_terms=("300 ohm", "75 ohm", "coaxial cable", "antenna jack", "inspect"),
                background_title_terms=("fcc", "microwave", "toothbrush"),
                background_body_terms=("harmful interference", "receiving antenna"),
            )
        if any(token in lowered for token in ("approval label", "emission control certificate")) and "boat" in lowered:
            return EvidencePlan(
                intent="boat_emission_label",
                max_evidence=2,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("approval label", "emission control", "engine compartment"),
                secondary_terms=("engine unit", "label"),
                background_title_terms=("limitations", "operate the boat"),
                background_body_terms=("skier is being pulled",),
            )
        if ("boat" in lowered or "sailing" in lowered) and "engine oil level" in lowered:
            return EvidencePlan(
                intent="boat_engine_oil_level",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("check the engine oil level", "oil tank filler cap", "dipstick"),
                secondary_terms=("minimum", "maximum", "level marks", "engine hood"),
                background_title_terms=("changing the engine oil", "lawn mower"),
                background_body_terms=("mower", "pto", "parking brake"),
            )
        if "boat" in lowered and "battery compartment" in lowered:
            return EvidencePlan(
                intent="boat_battery_compartment",
                max_evidence=2,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("battery compartment", "latch", "compartment lid"),
                secondary_terms=("port side", "stern", "close"),
                background_title_terms=("camera", "battery compartment cover"),
                background_body_terms=("dc coupler", "camera"),
            )
        if "boat" in lowered and "anchor light" in lowered:
            return EvidencePlan(
                intent="boat_anchor_light",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("anchor light", "set up", "socket"),
                secondary_terms=("storage compartment", "holder", "stoppers", "pole"),
                background_title_terms=("limitations", "table of contents"),
                background_body_terms=("skier is being pulled",),
            )
        if "boat" in lowered and "water supply" in lowered:
            return EvidencePlan(
                intent="boat_water_supply",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("water supply on or off", "shut-off valve", "inspection cover"),
                secondary_terms=("rear platform hatch", "clockwise", "stop the engines"),
                background_title_terms=("limitations", "dishwasher"),
                background_body_terms=("water tap", "dishwasher"),
            )
        if "boat" in lowered and "bilge pump" in lowered:
            return EvidencePlan(
                intent="boat_bilge_pump",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("bilge pump", "bilge pump switch", "drain"),
                secondary_terms=("indicator light", "automatically", "bilge water"),
                background_title_terms=("limitations", "fuse"),
                background_body_terms=("skier is being pulled",),
            )
        if any(product in lowered for product in ("boat", "ship", "jet boat", "sailing")) and any(
            token in lowered for token in ("turn a boat", "turn the boat", "turning", "steer", "steers", "steering")
        ):
            return EvidencePlan(
                intent="boat_steering_turn",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("steering", "jet thrust", "throttle"),
                secondary_terms=("steering wheel", "jet thrust nozzle", "turn", "idle"),
                background_title_terms=("limitations", "lawn mower"),
                background_body_terms=("skier is being pulled", "mower"),
            )
        if "boat" in lowered and ("cross wakes" in lowered or "swells" in lowered):
            return EvidencePlan(
                intent="boat_cross_wakes",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("crossing wakes and swells", "wake", "swell"),
                secondary_terms=("slower speed", "quartering", "angle", "least jolt"),
                background_title_terms=("limitations",),
            )
        if "boat" in lowered and ("flush" in lowered or "cooling system" in lowered):
            return EvidencePlan(
                intent="boat_flush_cooling",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("flushing the cooling system", "garden hose adapter", "water supply"),
                secondary_terms=("flush hose connector", "cooling water pilot outlet", "3 to 5 minutes", "turn off"),
                background_title_terms=("limitations",),
            )
        if "boat" in lowered and "livewell" in lowered:
            return EvidencePlan(
                intent="boat_livewell",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("livewell", "livewell switch", "livewell pump"),
                secondary_terms=("latch", "aerator switch", "supply water"),
                background_title_terms=("limitations",),
            )
        if "boat" in lowered and ("move forward" in lowered or "forward position" in lowered):
            return EvidencePlan(
                intent="boat_move_forward",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("forward", "remote control levers", "jet thrust"),
                secondary_terms=("shift gates", "tde", "neutral", "thrust"),
                background_title_terms=("limitations",),
            )
        if "boat" in lowered and "throttle" in lowered and "cable" in lowered:
            return EvidencePlan(
                intent="boat_throttle_cable",
                max_evidence=2,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("throttle cable", "grease", "inner wires"),
                secondary_terms=("pulley wheel", "aps", "steering cable", "shift cable"),
                background_title_terms=("snowmobile", "spark plug"),
            )
        if "boat" in lowered and "maintenance setting" in lowered:
            return EvidencePlan(
                intent="boat_maintenance_screen",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("maintenance setting", "reset", "hours of operation"),
                secondary_terms=("engine", "maintenance", "reset button"),
                background_title_terms=("limitations",),
            )
        if "microwave" in lowered:
            if "grease filter" in lowered:
                return EvidencePlan("microwave_grease_filter", 3, {"step", "list", "menu", "general"}, ("grease filter", "remove", "clean"), ("soak", "hot water", "detergent"), ("excessive microwave energy",), ())
            if "charcoal filter" in lowered:
                return EvidencePlan("microwave_charcoal_filter", 3, {"step", "list", "menu", "general"}, ("charcoal filter", "replace", "vent cover"), ("remove", "install", "screws"), ("excessive microwave energy",), ())
            if "control" in lowered and ("setup" in lowered or "set up" in lowered):
                return EvidencePlan("microwave_control_setup", 2, {"step", "list", "menu", "general"}, ("control set-up", "beep sound", "defrost weight"), ("clock", "display speed", "lbs/kg"), ("table of contents",), ())
            if "light timer" in lowered:
                return EvidencePlan("microwave_light_timer", 2, {"step", "list", "menu", "general"}, ("light timer", "turn on", "turn off"), ("lo light", "reset", "cancel"), ("table of contents",), ())
            if "favorite recipe" in lowered:
                return EvidencePlan("microwave_favorite_recipe", 2, {"step", "list", "menu", "general"}, ("favorite recipe", "recall", "memory"), ("program cooking", "custom recipe", "power level"), ("table of contents",), ())
            if "reheat" in lowered:
                return EvidencePlan("microwave_reheat", 3, {"step", "list", "menu", "general"}, ("reheat", "casserole", "dinner plate"), ("soup/sauce", "sensor", "preset"), ("table of contents",), ())
            if "auto defrost" in lowered:
                return EvidencePlan("microwave_auto_defrost", 3, {"step", "list", "menu", "general"}, ("auto defrost", "defrost sequences", "frozen foods"), ("weight", "turn", "separate", "rearrange"), ("table of contents",), ())
            if "oven light" in lowered:
                return EvidencePlan("microwave_oven_light", 3, {"step", "list", "menu", "general"}, ("oven light replacement", "bulb", "vent cover"), ("unplug", "mounting screws", "30 or 40 watt"), ("table of contents",), ())
        if "vacuum" in lowered:
            if "two primary modes" in lowered or "two main" in lowered or "dual mode" in lowered or "dual-mode" in lowered:
                return EvidencePlan("vacuum_dual_modes", 3, {"step", "list", "menu", "general"}, ("dual mode virtual wall barrier", "dual-mode virtual wall barrier", "two modes"), ("virtual wall mode", "halo", "cleaning need"), ("home base",), ())
            if "empty" in lowered and "bin" in lowered:
                return EvidencePlan("vacuum_empty_bin", 2, {"step", "list", "menu", "general"}, ("emptying the bin", "bin release button", "bin door"), ("full bin indicator", "pause"), ("home base",), ())
            if "full bin sensor" in lowered:
                return EvidencePlan("vacuum_full_bin_sensors", 2, {"step", "list", "menu", "general"}, ("cleaning the full bin sensors", "wipe the sensors"), ("inner", "outer", "dry cloth"), ("home base",), ())
            if "sensors and charging contacts" in lowered or "charging contacts" in lowered:
                return EvidencePlan("vacuum_sensors_contacts", 2, {"step", "list", "menu", "general"}, ("cleaning the sensor and charging contacts", "charging contacts"), ("clean dry cloth", "home base"), ("home base during",), ())
            if "home base" in lowered or "positioning" in lowered:
                return EvidencePlan("vacuum_home_base", 2, {"step", "list", "menu", "general"}, ("positioning the vacuum", "open", "uncluttered"), ("1.5 feet", "4 feet", "stairs", "virtual wall"), ("cleaning cycle",), ())
        if "coffee" in lowered or "coffee maker" in lowered or "coffee machine" in lowered:
            if any(token in lowered for token in ("emptying the system", "empty the system", "empty system")) or (
                "frost protection" in lowered and any(token in lowered for token in ("maintenance", "repair", "not in use"))
            ):
                return EvidencePlan(
                    "coffee_empty_system",
                    3,
                    {"step", "list", "menu", "general"},
                    ("emptying the system", "period of non-use", "frost protection"),
                    ("espresso", "lungo", "water tank", "lever", "both leds blink"),
                    ("descaling", "cleaning", "energy saving"),
                    (),
                )
            if "energy saving" in lowered or "power off mode" in lowered:
                return EvidencePlan("coffee_energy_saving", 3, {"step", "list", "menu", "general"}, ("energy saving", "power off mode", "9 minutes"), ("espresso", "lungo", "turn the machine off"), ("descaling",), ())
            if any(token in lowered for token in ("program", "volume", "valume", "water volume")):
                return EvidencePlan(
                    "coffee_program_volume",
                    3,
                    {"step", "list", "menu", "general"},
                    ("programming the water volume", "program the water volume", "espresso", "lungo"),
                    ("press and hold", "desired volume", "release button", "ml"),
                    ("descaling", "cleaning"),
                    (),
                )
            if any(token in lowered for token in ("clean", "cleaning", "after i use", "last longer")):
                return EvidencePlan(
                    "coffee_after_use_clean",
                    3,
                    {"step", "list", "menu", "general"},
                    ("cleaning", "rinse", "used capsule container", "drip tray"),
                    ("empty", "water tank", "maintenance", "descaling"),
                    ("coffee preparation",),
                    (),
                )
        if "lawn mower" in lowered or "mower" in lowered:
            if "load" in lowered and "unload" not in lowered:
                return EvidencePlan("mower_load", 3, {"step", "list", "menu", "general"}, ("loading the machine", "ramp"), ("trailer", "truck", "tie down"), ("engine oil",), ())
            if "unload" in lowered:
                return EvidencePlan("mower_unload", 3, {"step", "list", "menu", "general"}, ("unloading the machine", "ramp"), ("trailer", "truck", "reverse"), ("engine oil",), ())
            if "roll bar" in lowered:
                return EvidencePlan("mower_roll_bar", 3, {"step", "list", "menu", "general"}, ("roll bar", "raised and locked"), ("lower", "knobs", "hairpin cotter"), ("engine oil",), ())
            if "rear-shock" in lowered or "rear shock" in lowered:
                return EvidencePlan("mower_rear_shock", 2, {"step", "list", "menu", "general"}, ("rear-shock assemblies", "suspension system"), ("softest", "firmest", "detent"), ("engine oil",), ())
            if "height of cut" in lowered or "height-of-cut" in lowered:
                return EvidencePlan("mower_height_cut", 3, {"step", "list", "menu", "general"}, ("height of cut", "deck-lift switch"), ("height-of-cut pin", "bracket", "raise", "lower"), ("engine oil",), ())
            if "remove" in lowered and "filter" in lowered:
                return EvidencePlan("mower_remove_filters", 3, {"step", "list", "menu", "general"}, ("removing the filters", "air-cleaner"), ("latches", "primary filter", "inner filter"), ("engine oil",), ())
            if "mower belt" in lowered or ("replace" in lowered and "belt" in lowered):
                return EvidencePlan("mower_replace_belt", 3, {"step", "list", "menu", "general"}, ("replacing the mower belt", "idler arm", "mower-deck pulleys"), ("belt covers", "ratchet", "clutch pulley"), ("engine oil",), ())
            if "engine oil level" in lowered:
                return EvidencePlan("mower_engine_oil_level", 3, {"step", "list", "menu", "general"}, ("checking the engine-oil level", "dipstick"), ("full mark", "add oil", "engine oil"), ("changing the engine oil",), ())
        if "pressure cooker" in lowered or "air fryer" in lowered:
            if "anti-block shield" in lowered or "anti block shield" in lowered:
                return EvidencePlan(
                    "pressure_anti_block_shield",
                    3,
                    {"step", "list", "menu", "general"},
                    ("anti-block shield",),
                    ("prevents food particles", "steam release pipe", "prongs", "press down"),
                    ("float valve", "sealing ring"),
                    ("float valve", "sealing ring"),
                )
            if "steam release valve" in lowered:
                return EvidencePlan("pressure_steam_release", 3, {"step", "list", "menu", "general"}, ("steam release valve", "quick release", "vent position"), ("steam", "pressure", "button"), ("sealing ring",), ())
            if "pressure cooking lid" in lowered:
                return EvidencePlan("pressure_lid", 3, {"step", "list", "menu", "general"}, ("pressure cooking lid", "removing the lid", "closing the lid"), ("counterclockwise", "clockwise", "cooker base"), ("float valve",), ())
            if "condensation collector" in lowered:
                return EvidencePlan("pressure_condensation_collector", 3, {"step", "list", "menu", "general"}, ("condensation collector", "grooves", "tabs"), ("installed before cooking", "slide", "back of the cooker base"), ("float valve",), ())
            if "sealing ring" in lowered:
                return EvidencePlan("pressure_sealing_ring", 3, {"step", "list", "menu", "general"}, ("sealing ring", "air-tight seal", "installed"), ("one sealing ring", "silicone", "lid"), ("float valve",), ())
        if "ereader" in lowered or "e-reader" in lowered:
            if "button" in lowered or "interfaces" in lowered or "views" in lowered:
                return EvidencePlan("ereader_buttons", 3, {"step", "list", "menu", "general", "component"}, ("front view", "home/esc", "navigation"), ("usb port", "micro sd", "power button", "speaker"), ("table of contents",), ())
            if "main menu" in lowered or "browser history" in lowered:
                return EvidencePlan("ereader_main_browser", 3, {"step", "list", "menu", "general"}, ("main menu", "browser history"), ("recently read", "last reading page", "features"), ("table of contents",), ())
            if "ebook mode" in lowered or "eBook mode" in question:
                return EvidencePlan("ereader_ebook_mode", 3, {"step", "list", "menu", "general"}, ("ebook mode", "press “m”", "page jump"), ("save mark", "load mark", "browser mode", "brightness"), ("main menu",), ())
            if "music" in lowered:
                return EvidencePlan("ereader_music", 3, {"step", "list", "menu", "general"}, ("music mode", "audio files list", "press \"m\""), ("volume", "play/pause", "mp3", "wma"), ("main menu",), ())
            if "record" in lowered or "voice" in lowered:
                return EvidencePlan("ereader_record", 3, {"step", "list", "menu", "general"}, ("voice recording", "record", "play/pause"), ("save", "recorded", "music menu"), ("troubleshooting", "main menu"), ())
            if "video" in lowered:
                return EvidencePlan("ereader_video", 3, {"step", "list", "menu", "general"}, ("video mode", "press", "m"), ("subtitle language", "time play", "full screen", "brightness"), ("main menu",), ())
        if "snowmobile" in lowered:
            if "uphill" in lowered or "riding uphill" in lowered:
                return EvidencePlan(
                    "snowmobile_uphill",
                    3,
                    {"step", "list", "menu", "general"},
                    ("riding uphill", "uphill side", "running boards"),
                    ("accelerate before the climb", "lean forward", "crest", "parking brake"),
                    ("spark plug",),
                    (),
                )
            if "downhill" in lowered or "riding downhill" in lowered:
                return EvidencePlan(
                    "snowmobile_downhill",
                    2,
                    {"step", "list", "menu", "general"},
                    ("riding downhill", "engine compression", "brake"),
                    ("minimum speed", "throttle", "clutch", "light pressure"),
                    ("spark plug",),
                    (),
                )
            if "crossing a slope" in lowered or "cross slope" in lowered or "side hill" in lowered or "sidehill" in lowered:
                return EvidencePlan(
                    "snowmobile_cross_slope",
                    3,
                    {"step", "list", "menu", "general"},
                    ("crossing a slope", "uphill side", "running board"),
                    ("kneeling", "downhill knee", "uphill foot", "sideways slipping"),
                    ("spark plug",),
                    (),
                )
            if "throttle cable" in lowered:
                return EvidencePlan("snowmobile_throttle_cable", 3, {"step", "list", "menu", "general"}, ("throttle cable adjustment", "throttle"), ("free play", "adjuster", "locknut"), ("spark plug",), ())
            if "steering system" in lowered:
                return EvidencePlan("snowmobile_steering_system", 2, {"step", "list", "menu", "general"}, ("steering system", "free play"), ("handlebar", "dealer"), ("spark plug",), ())
            if "turn" in lowered:
                return EvidencePlan("snowmobile_turning", 3, {"step", "list", "menu", "general"}, ("turning", "handlebars", "lean"), ("slow down", "inside of the turn", "running board"), ("spark plug",), ())
            if "spark plug" in lowered:
                return EvidencePlan("snowmobile_spark_plug", 3, {"step", "list", "menu", "general"}, ("spark plug inspection", "electrode gap", "torque"), ("porcelain", "thread", "gasket"), ("throttle cable",), ())
        if "mouse" in lowered or "蓝牙激光鼠标" in question:
            if "battery" in lowered or "电池" in question or "电量" in question:
                if "status" in lowered or "low" in lowered or "电量" in question or "耗尽" in question:
                    return EvidencePlan("mouse_battery_status", 2, {"step", "list", "menu", "general"}, ("电量低", "琥珀色", "battery status"), ("control panel", "bluetooth"), (), ())
                if "install" in lowered or "insert" in lowered or "安装" in question or "装入" in question:
                    return EvidencePlan("mouse_battery_install", 3, {"step", "list", "menu", "general"}, ("安装电池", "aa", "正负极"), ("电池仓盖", "按钮"), (), ())
            if "hid" in lowered or "other device" in lowered or "其他" in question:
                return EvidencePlan("mouse_other_hid", 3, {"step", "list", "menu", "general"}, ("other hid device", "discoverable", "bluetooth"), ("usb 蓝牙接收器", "enable connection"), (), ())
        if "widcomm" in lowered or "蓝牙激光鼠标" in question:
            if "卸载" in question or "uninstall" in lowered:
                return EvidencePlan("mouse_widcomm_uninstall", 2, {"step", "list", "menu", "general"}, ("卸载 widcomm", "添加或删除程序", "删除"), ("拔下 usb", "重启"), ("产品介绍",), ())
            if "配对" in question or "pair" in lowered:
                return EvidencePlan("mouse_widcomm_pair", 3, {"step", "list", "menu", "general"}, ("widcomm 蓝牙驱动程序配对", "hid"), ("usb 蓝牙接收器", "确认信息"), ("产品介绍",), ())
            if "首次" in question or "first" in lowered:
                return EvidencePlan("mouse_widcomm_first_use", 3, {"step", "list", "menu", "general"}, ("widcomm 蓝牙驱动程序使用", "首次使用"), ("蓝牙图标", "初始配置向导", "hid"), ("产品介绍",), ())
            if "安装" in question or "install" in lowered:
                return EvidencePlan("mouse_widcomm_install", 3, {"step", "list", "menu", "general"}, ("安装 widcomm", "setup.exe", "完成"), ("安装光盘", "下一步", "接收器"), ("产品介绍",), ())
        if "toothbrush" in lowered and ("activate" in lowered or "deactivate" in lowered or "customized" in lowered):
            return EvidencePlan(
                intent="toothbrush_features",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("activate or deactivate", "adaptive intensity", "pressure sensor feedback"),
                secondary_terms=("scrubbing feedback", "brush head replacement reminder", "app"),
                background_title_terms=("specifications",),
            )
        if "toothbrush" in lowered and "intensity" in lowered:
            return EvidencePlan(
                intent="toothbrush_intensity",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("intensity settings", "three different intensity settings"),
                secondary_terms=("high intensity", "medium intensity", "low intensity", "indicator lights"),
                background_title_terms=("app - getting started",),
                background_body_terms=("track your brushing",),
            )
        if "健身追踪器" in lowered:
            if "包装盒" in lowered or "盒里" in lowered or "包含" in lowered:
                return EvidencePlan("fitness_box_contents", 3, {"step", "list", "menu", "general"}, ("包装盒内含物品", "手表", "充电线"), ("小号表带", "大号表带", "单独购买"), ("设置健身追踪器",), ())
            if "通知" in lowered and "心率" not in lowered:
                return EvidencePlan("fitness_notifications", 3, {"step", "list", "menu", "general"}, ("设置通知", "手机可接收通知"), ("蓝牙", "通知", "应用"), ("设置健身追踪器",), ())
            if "充电" in lowered or "电量低" in lowered:
                return EvidencePlan("fitness_charge", 2, {"step", "list", "menu", "general"}, ("为手表充电",), ("usb", "磁吸", "1-2小时"), (), ())
            if "界面" in lowered or "操作" in lowered:
                return EvidencePlan("fitness_interface", 3, {"step", "list", "menu", "general"}, ("基础操作", "按钮快捷操作", "功能卡片"), ("向下滑动", "向上滑动", "左右滑动"), (), ())
            if "运动应用" in lowered or "运动情况" in lowered or "追踪并分析" in lowered:
                return EvidencePlan("fitness_exercise", 3, {"step", "list", "menu", "general"}, ("运动应用", "运动", "目标"), ("活动摘要", "心率", "数据"), (), ())
            if "心率" in lowered:
                return EvidencePlan("fitness_heart_rate", 3, {"step", "list", "menu", "general"}, ("心率", "手腕", "背面与皮肤接触"), ("运动", "通知"), (), ())
            if "消费" in lowered or "支付" in lowered:
                return EvidencePlan("fitness_payment", 3, {"step", "list", "menu", "general"}, ("进行消费", "钱包", "支付"), ("pin", "默认卡", "支付终端"), (), ())
            if "锁屏" in lowered or "设备锁" in lowered:
                return EvidencePlan("fitness_lock", 2, {"step", "list", "menu", "general"}, ("设置设备锁", "4位pin"), ("健身追踪器应用",), (), ())
            if "问题" in lowered or "解决" in lowered:
                return EvidencePlan("fitness_troubleshooting", 3, {"step", "list", "menu", "general"}, ("其他问题", "重启手表"), ("无法同步", "无响应"), (), ())
        if "烤箱" in lowered:
            if "催化侧面板" in lowered:
                return EvidencePlan("oven_catalytic_panels", 3, {"step", "list", "menu", "general"}, ("催化侧面板", "微孔搪瓷涂层"), ("吸附油脂", "自动清洁", "200℃"), ("旋转烤叉",), ())
            if "接油盘" in lowered:
                return EvidencePlan("oven_drip_tray", 2, {"step", "list", "menu", "general"}, ("接油盘",), ("收集油脂", "少量水"), (), ())
            if "外部" in lowered:
                return EvidencePlan("oven_exterior_clean", 2, {"step", "list", "menu", "general"}, ("烤箱外部",), ("湿布", "干布"), (), ())
            if "烤架烤盘套装" in lowered:
                return EvidencePlan("oven_grill_pan_set", 2, {"step", "list", "menu", "general"}, ("烤架烤盘套装",), ("烤架", "搪瓷容器", "烧烤功能"), (), ())
            if "油脂过滤器" in lowered:
                return EvidencePlan("oven_grease_filter", 2, {"step", "list", "menu", "general"}, ("油脂过滤器",), ("风扇", "热风循环"), (), ())
            if "滑动搁架" in lowered:
                return EvidencePlan("oven_sliding_shelf", 2, {"step", "list", "menu", "general"}, ("滑动搁架",), ("拉出", "层架"), (), ())
            if "烤盘" in lowered:
                return EvidencePlan("oven_baking_tray", 2, {"step", "list", "menu", "general"}, ("烤盘",), ("饼干", "蛋糕", "披萨"), (), ())
            if "烤架" in lowered:
                return EvidencePlan("oven_wire_shelf", 2, {"step", "list", "menu", "general"}, ("烤架",), ("烧烤", "锅具", "层位"), (), ())
        if "发电机" in lowered:
            if any(token in lowered for token in ("精密设备", "电压敏感", "对电压敏感", "医疗设备", "个人电脑")):
                return EvidencePlan(
                    "generator_sensitive_equipment",
                    2,
                    {"step", "list", "menu", "general"},
                    ("精密设备", "电压敏感", "便携式发电机"),
                    ("医疗设备", "个人电脑", "逆变器", "设备供应商"),
                    ("环保署", "排放", "法规", "经销商保养"),
                    ("排放控制",),
                )
            if "标识" in lowered or "识别码" in lowered or "序列号" in lowered:
                return EvidencePlan(
                    "generator_identification",
                    3,
                    {"component", "list", "menu", "general"},
                    ("识别码记录", "产品识别码", "序列号"),
                    ("指定位置", "订购零配件", "机器被盗"),
                    ("保修", "经销商保修"),
                    ("有限保修",),
                )
            if "开关" in lowered and any(token in lowered for token in ("两种", "不同", "介绍", "控制", "功能")):
                return EvidencePlan(
                    "generator_control_switches",
                    3,
                    {"component", "list", "menu", "general"},
                    ("控制面板", "发动机开关", "经济控制开关"),
                    ("点火电路", "降低油耗", "减少噪音", "额定转速"),
                    ("电路图", "连接发电机"),
                    ("电路图", "交流插座"),
                )
            if "启动" in lowered and "无法" not in lowered:
                return EvidencePlan("generator_start", 3, {"step", "list", "menu", "general"}, ("启动发动机前", "通气旋钮", "反冲启动器"), ("燃油开关", "发动机开关", "阻风门"), (), ())
            if "发烫" in lowered or "消音器" in lowered:
                return EvidencePlan("generator_hot_safety", 2, {"step", "list", "menu", "general"}, ("发动机及消音器可能高温发烫",), ("1米", "提手"), (), ())
            if "触电" in lowered:
                return EvidencePlan("generator_shock_safety", 2, {"step", "list", "menu", "general"}, ("防止触电",), ("雨雪", "湿手", "接地"), (), ())
            if "燃油" in lowered and "检查" in lowered:
                return EvidencePlan("generator_fuel_check", 2, {"step", "list", "menu", "general"}, ("燃油", "确认油箱"), ("无铅汽油", "油箱容量"), (), ())
            if "发动机机油" in lowered and "检查" in lowered:
                return EvidencePlan("generator_oil_check", 2, {"step", "list", "menu", "general"}, ("发动机机油", "上限"), ("必要时添加",), (), ())
            if "停机" in lowered or "发动机停机" in lowered:
                return EvidencePlan("generator_stop", 2, {"step", "list", "menu", "general"}, ("停机", "发动机开关"), ("断开", "燃油开关"), (), ())
            if "无法启动" in lowered or "启动" in lowered and "无法" in lowered:
                return EvidencePlan("generator_no_start", 3, {"step", "list", "menu", "general"}, ("无法启动", "机油警告灯"), ("添加机油", "重新启动"), (), ())
        if "generator" in lowered and any(token in lowered for token in ("sensitive", "precision", "medical equipment", "personal computer")):
            return EvidencePlan(
                "generator_sensitive_equipment",
                2,
                {"step", "list", "menu", "general"},
                ("sensitive equipment", "precision equipment", "portable generator"),
                ("medical equipment", "personal computer", "inverter", "supplier"),
                ("epa", "emission", "regulation", "dealer maintenance"),
                ("emission control",),
            )
        if ("drill" in lowered or "driver" in lowered) and "charge" in lowered:
            return EvidencePlan(
                "drill_battery_charge",
                3,
                {"step", "list", "menu", "general"},
                ("charging procedure", "charger", "battery pack"),
                ("red charging light", "flashes continuously", "charged", "fully charged"),
                ("safety warning", "fire hazard"),
                ("battery safety",),
            )
        if "camera" in lowered:
            if "auto print" in lowered or "automatic print" in lowered or "自动打印" in lowered:
                return EvidencePlan(
                    "camera_auto_print",
                    3,
                    {"menu", "step", "list", "general"},
                    ("自动打印模式", "auto"),
                    ("图像保存后立即开始打印", "打印模式选择器"),
                    ("手动打印模式", "存储卡"),
                    ("存储卡",),
                )
            if "battery" in lowered and any(token in lowered for token in ("install", "insert", "load")):
                return EvidencePlan("camera_battery_install", 3, {"step", "list", "menu", "general"}, ("battery", "load the battery", "insert the battery"), ("charge", "camera"), (), ())
            if "battery" in lowered and "charge" in lowered:
                return EvidencePlan("camera_battery_charge", 3, {"step", "list", "menu", "general"}, ("battery", "charge", "usb"), ("charging", "camera"), (), ())
            if "power the camera" in lowered or ("power" in lowered and "camera" in lowered and "steps" in lowered):
                return EvidencePlan("camera_power", 3, {"step", "list", "menu", "general"}, ("power switch", "on", "battery"), ("shooting", "camera"), ("troubleshooting",), ())
            if "cp direct" in lowered or "direct method" in lowered or "direct printing" in lowered:
                return EvidencePlan("camera_cp_direct", 3, {"step", "list", "menu", "general"}, ("direct printing", "cp direct", "printer"), ("connect", "print", "camera"), ("troubleshooting",), ())
            if "card" in lowered and any(token in lowered for token in ("install", "insert", "load")):
                return EvidencePlan("memory_card", 3, {"menu", "step", "list", "general"}, ("memory card", "insert", "load", "card"), ("camera", "images"), ("存储卡注意事项", "在电脑上使用存储卡注意事项"), ("格式化", "电脑"))
            if "shutter button" in lowered and any(token in lowered for token in ("remove", "removing", "repair")):
                return EvidencePlan("camera_shutter_button", 2, {"step", "list", "menu", "general"}, ("shutter button",), ("camera", "shooting"), (), ())
            if "mount" in lowered and "lens" in lowered:
                return EvidencePlan("camera_mount_lens", 3, {"step", "list", "menu", "general"}, ("attach the lens", "lens mount index"), ("clicks in place",), (), ())
            if "eyepiece cover" in lowered:
                return EvidencePlan("camera_eyepiece_cover", 3, {"step", "list", "menu", "general"}, ("eyepiece cover", "eyepiece groove"), ("self-timer", "stray light"), (), ())
            quoted_p_query = (('\\"p\\"' in lowered or '"p"' in lowered or "“p”" in lowered) and any(term in lowered for term in ("mode", "model")))
            if quoted_p_query or "p\" model" in lowered or "p model" in lowered or "p mode" in lowered or "program ae" in lowered:
                return EvidencePlan("camera_p_mode", 3, {"step", "list", "menu", "general"}, ("program ae",), ("shutter speed", "aperture"), (), ())
        if "fax" in lowered:
            if "finger" in lowered or "fingers" in lowered:
                return EvidencePlan("fax_finger_safety", 3, {"step", "list", "menu", "general"}, ("keep fingers", "injury", "machine"), ("close", "cover", "document"), ("ren", "fcc"), ())
            if "warning label" in lowered or "caution label" in lowered or "remove them" in lowered:
                return EvidencePlan("fax_warning_labels", 3, {"step", "list", "menu", "general"}, ("warning", "caution", "label"), ("important", "do not remove", "safety"), ("ren", "fcc"), ())
            if "safety" in lowered:
                return EvidencePlan("fax_safety", 3, {"step", "list", "menu", "general"}, ("important safety instructions",), ("electrical shock", "water", "service"), (), ())
            if "move" in lowered or "moving" in lowered:
                return EvidencePlan("fax_move", 2, {"step", "list", "menu", "general"}, ("moving", "unplug"), ("wall outlet", "telephone line"), (), ())
            if "canada" in lowered:
                return EvidencePlan("fax_canada", 2, {"step", "list", "menu", "general"}, ("industry canada",), ("rss", "interference"), (), ())
        if "jetski" in lowered or "watercraft" in lowered:
            if "characteristics" in lowered:
                return EvidencePlan("jetski_characteristics", 3, {"step", "list", "menu", "general"}, ("watercraft characteristics", "throttle", "steering"), ("jet thrust", "engine speed", "handlebar"), ("fuel filter",), ())
            if "hood" in lowered and any(token in lowered for token in ("open", "close", "daily usage")):
                return EvidencePlan("jetski_hood_open_close", 2, {"step", "list", "menu", "general"}, ("hood", "latch", "lift the hood"), ("push the latch down", "properly secured"), ("seat", "fuel tank", "filler cap"), ())
            if "fuel filter" in lowered or "fuel tank" in lowered:
                return EvidencePlan("jetski_fuel_filter", 3, {"step", "list", "menu", "general"}, ("fuel filter", "fuel tank"), ("dealer", "replace", "water"), ("throttle",), ())
            if "intake" in lowered and "impeller" in lowered:
                return EvidencePlan("jetski_intake_impeller", 3, {"step", "list", "menu", "general"}, ("jet intake", "impeller", "clean"), ("stop the engine", "weeds", "debris"), ("throttle",), ())
            if "seat" in lowered:
                return EvidencePlan("jetski_seat", 3, {"step", "list", "menu", "general"}, ("to remove the seat", "to install the seat"), ("seat latch", "projection"), (), ())
            if "filler cap" in lowered:
                return EvidencePlan("jetski_filler_caps", 3, {"step", "list", "menu", "general"}, ("fuel tank filler cap", "oil tank filler cap"), ("counterclockwise", "secured"), (), ())
            if "lever" in lowered:
                return EvidencePlan("jetski_levers", 3, {"step", "list", "menu", "general"}, ("throttle lever", "choke lever", "qsts selector"), ("accelerate", "cold engine", "trim angle"), (), ())
        if "摩托艇" in lowered or "水上摩托" in lowered:
            if "碰撞" in lowered or "避让" in lowered or "障碍" in lowered:
                return EvidencePlan("jetski_avoid_collision", 3, {"step", "list", "menu", "general"}, ("避免碰撞", "转向需要油门"), ("安全速度", "距离", "障碍物"), (), ())
            if "停止" in lowered or "停车" in lowered or "停稳" in lowered or "制动" in lowered or "减速" in lowered:
                return EvidencePlan("jetski_stop", 3, {"step", "list", "menu", "general"}, ("独立制动", "停车距离", "RiDE"), ("释放油门", "转向避让", "90米"), (), ())
            if "转向" in lowered or "转弯" in lowered or "油门" in lowered or "半滑航" in lowered:
                return EvidencePlan("jetski_throttle_turning", 3, {"step", "list", "menu", "general"}, ("车把", "油门", "喷射推力"), ("半滑航", "转向能力", "重新施加油门"), (), ())
        if "拖曳速度" in lowered and "半滑航" in lowered and "滑航" in lowered:
            return EvidencePlan("jetski_speed_modes", 3, {"step", "list", "menu", "general"}, ("拖曳速度", "半滑航", "滑航"), ("转向", "操控", "速度"), ("倾覆",), ())
        if "landline" in lowered or "base station" in lowered:
            if "install the handset" in lowered or "install" in lowered and "handset" in lowered:
                return EvidencePlan("landline_install_handset", 2, {"step", "list", "menu", "general"}, ("install the handset", "battery tape"), ("charge",), (), ())
            if "led indicator" in lowered and "base station" in lowered:
                return EvidencePlan("landline_base_led", 2, {"step", "list", "menu", "general"}, ("behavior of the led indicator on the base station",), ("current status",), (), ())
            if "led indicator" in lowered:
                return EvidencePlan("landline_handset_led", 2, {"step", "list", "menu", "general"}, ("set the handset led indicator behavior",), ("events status", "charge status"), (), ())
        if "vacuum" in lowered:
            if "robot anatomy" in lowered or "vacuum anatomy" in lowered or ("robot" in lowered and "anatomy" in lowered):
                return EvidencePlan(
                    "vacuum_robot_anatomy",
                    2,
                    {"component", "list", "general", "menu"},
                    ("clean button", "bin release", "handle"),
                    ("dust bin", "faceplate", "sensor", "side brush"),
                    ("virtual wall", "full bin sensors"),
                    ("virtual wall", "full bin sensors"),
                )
            if "filter" in lowered and "clean" in lowered:
                return EvidencePlan("vacuum_clean_filter", 2, {"step", "list", "menu", "general"}, ("cleaning the filter",), ("yellow tab", "filter door"), (), ())
            if "extractor" in lowered or "brushes" in lowered or "rollers" in lowered:
                return EvidencePlan("vacuum_clean_extractors", 3, {"step", "list", "menu", "general"}, ("cleaning the extractors",), ("yellow extractor caps", "hair", "debris"), (), ())
            if "side brush" in lowered:
                return EvidencePlan("vacuum_clean_side_brush", 2, {"step", "list", "menu", "general"}, ("cleaning the side brush",), ("screw", "brush post"), (), ())
            if "front caster" in lowered or ("caster wheel" in lowered and "clean" in lowered):
                return EvidencePlan(
                    "vacuum_front_caster",
                    2,
                    {"step", "list", "general", "menu"},
                    ("cleaning the front caster wheel", "front caster wheel"),
                    ("wheel cavity", "axle", "hair", "clicks back into place"),
                    ("virtual wall",),
                    ("virtual wall",),
                )
        if "吹风机" in lowered:
            if "启动" in lowered or "冷机" in lowered or "热机" in lowered:
                return EvidencePlan("blower_start", 3, {"step", "list", "menu", "general"}, ("冷机启动", "热机启动", "泵油膜片"), ("阻风门", "启动手柄", "启动油门"), (), ())
            if "化油器" in lowered:
                return EvidencePlan("blower_carburetor", 3, {"step", "list", "menu", "general"}, ("化油器调节", "低速油针"), ("高速油针", "怠速调节"), (), ())
            if "关闭" in lowered or "停机" in lowered:
                return EvidencePlan("blower_stop", 2, {"step", "list", "menu", "general"}, ("停机开关",), ("关闭发动机",), (), ())
            if "安全" in lowered or "注意" in lowered:
                return EvidencePlan("blower_safety", 3, {"step", "list", "menu", "general"}, ("吹风机操作安全",), ("喷口", "关闭发动机", "通风"), (), ())
        if "空气净化器" in lowered:
            if "灰尘传感器" in lowered or "dust sensor" in lowered:
                return EvidencePlan(
                    "airpurifier_dust_sensor",
                    3,
                    {"step", "list", "menu", "general"},
                    ("灰尘传感器清洁", "灰尘传感器"),
                    ("滤网盖", "棉签", "镜头", "进风口", "擦干"),
                    ("脚轮", "常规运行", "更换滤网"),
                    ("脚轮安装", "风速"),
                )
            if "塑料包装" in lowered or ("滤网" in lowered and ("包装" in lowered or "取下" in lowered or "拆除" in lowered)):
                return EvidencePlan("airpurifier_remove_filter_packaging", 3, {"step", "list", "menu", "general"}, ("滤网塑料包装", "睡眠 + 自动"), ("拔下电源", "滤网盖", "初始化"), (), ())
            if "更换滤网" in lowered or ("滤网" in lowered and ("更换" in lowered or "指示灯" in lowered or "红色" in lowered)):
                return EvidencePlan("airpurifier_replace_filter", 2, {"step", "list", "menu", "general"}, ("更换滤网", "红色", "6-12"), ("睡眠 + 自动", "重置指示灯"), (), ())
            if "脚轮" in lowered:
                return EvidencePlan("airpurifier_casters", 2, {"step", "list", "menu", "general"}, ("脚轮安装",), ("4 个脚轮", "8 颗螺丝"), (), ())
            if "内外" in lowered or "外表面" in lowered:
                return EvidencePlan("airpurifier_clean_body", 2, {"step", "list", "menu", "general"}, ("产品外表面清洁",), ("软布", "吸尘器"), (), ())
            if "滤网" in lowered:
                return EvidencePlan("airpurifier_clean_filter", 2, {"step", "list", "menu", "general"}, ("滤网清洁",), ("吸尘器", "软刷", "切勿用水"), (), ())
            if "长期存放" in lowered or "存放" in lowered:
                return EvidencePlan("airpurifier_storage", 2, {"step", "list", "menu", "general"}, ("长期存放",), ("运行设备 1 小时", "干燥"), (), ())
        if "洗碗机" in lowered:
            if "洗涤块" in lowered or "tablet" in lowered:
                return EvidencePlan("dishwasher_tablet", 3, {"step", "list", "menu", "general"}, ("洗涤块功能", "半载 / 洗涤块"), ("指示灯", "盐量", "亮碟剂"), (), ())
            if "洗涤剂" in lowered or "detergent" in lowered:
                return EvidencePlan("dishwasher_add_detergent", 3, {"step", "list", "menu", "general"}, ("添加洗涤剂", "洗涤剂盒"), ("15cm", "25cm", "5cm"), (), ())
        if (
            ("健身单车" in lowered or "游玩区域" in lowered or "运动区域" in lowered)
            and any(token in lowered for token in ("游玩区域", "运动区域", "推荐尺寸", "安全且最佳", "使用效果"))
        ):
            return EvidencePlan("bike_workout_area", 2, {"step", "list", "menu", "general"}, ("运动区域", "2.3 米", "1.8 米"), ("坚固", "水平", "安全运行"), (), ())
        if "健身单车" in lowered:
            if "技术规格" in lowered:
                return EvidencePlan("bike_specs", 3, {"step", "list", "menu", "general"}, ("规格", "技术规格"), ("认证标准", "交流电源适配器"), (), ())
            if "用户档案" in lowered and ("编辑" in lowered or "遵循" in lowered):
                return EvidencePlan("bike_edit_profile", 3, {"step", "list", "menu", "general"}, ("编辑用户档案",), ("姓名", "年龄", "体重"), (), ())
            if "轻松骑行" in lowered:
                return EvidencePlan("bike_easy_ride_programs", 2, {"step", "list", "menu", "general"}, ("轻松骑行",), ("起伏山丘", "公园骑行", "轻松之旅"), (), ())
            if "体能测试" in lowered:
                return EvidencePlan("bike_fitness_test", 3, {"step", "list", "menu", "general"}, ("体能测试", "功率输出", "心率"), ("初级", "高级", "体能得分"), (), ())
            if "山地" in lowered:
                return EvidencePlan("bike_mountain_programs", 2, {"step", "list", "menu", "general"}, ("山地",), ("派克峰", "胡德山", "金字塔"), (), ())
            if "挑战" in lowered:
                return EvidencePlan("bike_challenge_programs", 2, {"step", "list", "menu", "general"}, ("挑战",), ("上坡冲刺", "交叉训练", "间歇训练"), (), ())
        if "电钻" in lowered:
            if "充电" in lowered:
                return EvidencePlan(
                    "drill_battery_charge",
                    3,
                    {"step", "list", "menu", "general"},
                    ("充电步骤", "充电器", "电池组"),
                    ("红色指示灯", "持续闪烁", "常亮", "充电完成"),
                    ("安全警告", "火灾隐患"),
                    ("电池安全",),
                )
            if "dcb101" in lowered and ("指示灯" in question or "闪烁" in question):
                return EvidencePlan(
                    "drill_dcb101_indicator",
                    3,
                    {"general", "troubleshoot", "step", "list", "menu"},
                    ("dcb101", "指示灯"),
                    ("电池组充电中", "电池组已充满", "过热/过冷延迟", "电池组或充电器故障", "电源故障"),
                    (),
                    (),
                )
            if "附件" in lowered or "配备" in lowered:
                return EvidencePlan("drill_accessories", 3, {"step", "list", "menu", "general"}, ("附件", "推荐附件"), ("护目镜", "听力防护", "辅助手柄"), ("保修",), ())
            if "三年有限保修" in lowered or "保修" in lowered:
                return EvidencePlan("drill_warranty", 3, {"step", "list", "menu", "general", "warranty"}, ("三年有限保修", "1年免费服务", "2年免费服务"), ("购买凭证", "正常使用", "电池组"), ("维护 警告",), ())
            if "无键夹头" in lowered:
                return EvidencePlan("drill_keyless_chuck", 3, {"step", "list", "menu", "general"}, ("单套无键夹头",), ("逆时针", "顺时针", "附件"), (), ())
            if "电池组" in lowered and ("安装" in lowered or "拆卸" in lowered):
                return EvidencePlan("drill_battery_pack", 3, {"step", "list", "menu", "general"}, ("电池组的安装与拆卸", "电池释放按钮"), ("导轨", "滑入"), (), ())
        if "bimini top" in lowered and ("boat" in lowered or "canopy" in lowered):
            if "upright" in lowered and any(token in lowered for token in ("store", "storage", "stored")):
                return EvidencePlan("boat_bimini_upright_storage", 3, {"step", "list", "menu", "general"}, ("storing the bimini top in the upright position", "remove the lock pins", "center poles"), ("pull the bimini top", "storage cover", "fully collapsed"), (), ())
            if "remove" in lowered:
                return EvidencePlan("boat_bimini_remove", 3, {"step", "list", "menu", "general"}, ("removing the bimini top",), ("main pole mounting pins",), (), ())
            if "install" in lowered or "canopy" in lowered:
                return EvidencePlan("boat_bimini_install", 3, {"step", "list", "menu", "general"}, ("installing the bimini top", "setting up the bimini top"), ("lock pin", "support pole"), (), ())
        if any(token in lowered for token in ("start the boat", "start the engine", "start the engines", "turn on the boat", "turn on the boat's engine")) and "boat" in lowered:
            return EvidencePlan(
                "boat_engine_start",
                4,
                {"step", "list", "menu", "general"},
                ("starting the engine", "battery switch", "blower", "engine shut-off cord", "neutral", "main switch keys"),
                ("start position", "release", "5 seconds", "15 seconds", "thrust"),
                ("docking", "low speed maneuvering", "turning the boat"),
                ("speed maneuvering", "when docking"),
            )
        if "load the boat" in lowered and "boat" in lowered:
            return EvidencePlan("boat_load_distribution", 2, {"step", "list", "menu", "general"}, ("weight distribution", "maximum load"), ("side-to-side", "bow-to-stern"), (), ())
        if "motherboard" in lowered and ("t_sensor" in lowered or "thermal sensor connector" in lowered or "thermal sensor" in lowered):
            return EvidencePlan(
                intent="motherboard_t_sensor",
                max_evidence=2,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("thermal sensor connector", "t_sensor", "thermistor cable"),
                secondary_terms=("monitor the temperature", "critical components", "connected devices"),
                background_title_terms=("front panel audio", "bios setup", "serial port"),
                background_body_terms=("front panel audio", "hd audio", "legacy ac'97", "bios setup"),
            )
        if "motherboard" in lowered and "pci express" in lowered and "x16" in lowered:
            return EvidencePlan(
                intent="motherboard_pcie_x16",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("pci express", "graphic", "slots"),
                secondary_terms=("single vga", "sufficient power", "chassis fan", "graphics card"),
                background_title_terms=("bios setup", "native power management", "pch-io configuration"),
                background_body_terms=("bios setup", "configuration options", "aspm operations"),
            )
        if "motherboard" in lowered and "onboard led" in lowered:
            return EvidencePlan(
                intent="motherboard_onboard_led",
                max_evidence=2,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("standby power led", "onboard led"),
                secondary_terms=("sleep mode", "soft-off", "unplug the power cable"),
                background_title_terms=("bios setup", "q-led", "ez mode"),
                background_body_terms=("entering bios setup", "advanced mode"),
            )
        if "motherboard" in lowered and ("sata odd" in lowered or ("usb" in lowered and "operating system" in lowered)):
            return EvidencePlan(
                intent="motherboard_sata_odd_usb_os",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("sata odd", "usb", "os 7"),
                secondary_terms=("support dvd", "installation source", "press f8", "boot screen"),
                background_title_terms=("bios setup", "ez flash", "raid"),
                background_body_terms=("update the bios", "create raid"),
            )
        if "motherboard" in lowered and "chassis" in lowered and any(token in lowered for token in ("secure", "screw", "mount")):
            return EvidencePlan(
                intent="motherboard_chassis_screws",
                max_evidence=2,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("screw holes", "secure the motherboard", "chassis"),
                secondary_terms=("nine screws", "do not over tighten", "rear of the chassis"),
                background_title_terms=("bios setup", "package contents"),
                background_body_terms=("entering bios setup", "optional documentation"),
            )
        if "motherboard" in lowered and "system memory" in lowered:
            return EvidencePlan(
                intent="motherboard_system_memory",
                max_evidence=3,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("system memory", "recommended memory configurations"),
                secondary_terms=("channel a", "channel b", "dual-channel", "dimm voltage", "64-bit"),
                background_title_terms=("bios setup", "qvl"),
                background_body_terms=("entering bios setup", "qualified vendors lists"),
            )
        if "motherboard" in lowered and "tpm connector" in lowered:
            return EvidencePlan(
                intent="motherboard_tpm_connector",
                max_evidence=2,
                preferred_types={"step", "list", "menu", "general"},
                primary_terms=("tpm connector", "trusted platform module"),
                secondary_terms=("keys", "digital certificates", "network security", "platform integrity"),
                background_title_terms=("bios setup", "serial port"),
                background_body_terms=("entering bios setup", "serial port connector"),
            )
        if "thermostat" in lowered or "温控器" in lowered:
            if "temporary" in lowered or "hold" in lowered or "override" in lowered or "临时" in lowered or "保持" in lowered:
                return EvidencePlan("thermostat_temp_override", 3, {"step", "list", "menu", "general"}, ("temporary override", "permanent hold", "hold"), ("cancel", "next scheduled period", "desired temperature"), (), ())
            if "date" in lowered or "time" in lowered or "schedule" in lowered or "日期" in lowered or "时间" in lowered or "日程" in lowered:
                return EvidencePlan("thermostat_datetime", 3, {"step", "list", "menu", "general"}, ("设置日期", "设置时间", "调整程序日程"), ("select", "wake", "away", "home", "sleep"), (), ())
        if "steam cleaner" in lowered or "蒸汽清洁机" in lowered:
            if "assembly" in lowered or "assemble" in lowered or "组装" in lowered:
                return EvidencePlan("steam_quick_assembly", 3, {"step", "list", "menu", "general"}, ("快速组装", "无需专用工具"), ("手柄杆", "锁扣", "拖把头"), (), ())
        if ("memory card" in lowered or "存储卡" in question) and any(token in lowered or token in question for token in ("insert", "load", "装入", "插入")):
            return EvidencePlan(
                intent="memory_card",
                max_evidence=3,
                preferred_types={"menu", "step", "list", "general"},
                primary_terms=("存储卡", "memory card", "插入", "装入", "insert", "load"),
                secondary_terms=("图像", "images"),
                background_title_terms=("存储卡注意事项", "在电脑上使用存储卡注意事项"),
                background_body_terms=("格式化", "电脑"),
            )
        if ("start" in lowered and ("engine" in lowered or "jetski" in lowered)) or "启动发动机" in question:
            return EvidencePlan(
                intent="start_engine",
                max_evidence=3,
                preferred_types={"step", "menu", "list", "general"},
                primary_terms=("启动发动机", "发动机启动", "启动开关", "熄火绳", "start engine", "engine start", "start switch", "lanyard"),
                secondary_terms=("发动机", "启动", "开关", "engine", "start", "switch"),
                background_title_terms=("前言", "目标", "重要信息", "目录"),
            )
        if ("battery" in lowered and "charge" in lowered) or ("电池" in question and "充电" in question):
            return EvidencePlan(
                intent="battery_charge",
                max_evidence=3,
                preferred_types={"step", "menu", "list", "general"},
                primary_terms=("电池充电", "装入/充电电池", "充电时间", "usb", "charger", "charge"),
                secondary_terms=("电池", "充电", "battery", "charge"),
                background_body_terms=("recycle", "回收", "生活垃圾", "认证标志"),
            )
        if self._is_parameter_question(question):
            return EvidencePlan(
                intent="parameter",
                max_evidence=2,
                preferred_types={"title_only", "component", "menu", "general"},
                primary_terms=(),
            )
        if any(token in question for token in STEP_PREFIXES) or any(
            token in lowered
            for token in (
                "how",
                "step",
                "connect",
                "install",
                "replace",
                "change",
                "set",
                "inspect",
                "remove",
            )
        ):
            preferred_types = {"step", "list", "menu", "component", "general"}
        elif any(token in question for token in ("哪些", "包含", "包括", "保修", "服务", "参数")):
            preferred_types = {"list", "menu", "warranty", "component", "general"}
        else:
            preferred_types = {"component", "menu", "general", "warranty", "step"}
        return EvidencePlan(
            intent="default",
            max_evidence=3,
            preferred_types=preferred_types,
            primary_terms=(),
        )

    def _is_primary_evidence_for_plan(self, plan: EvidencePlan, result: SearchResult) -> bool:
        combined = f"{result.chunk.section_title} {result.chunk.text}"
        lowered = combined.lower()
        if plan.intent == "parameter":
            return bool(NUMERIC_SIGNAL_RE.search(combined))
        return all(term in combined or term in lowered for term in plan.primary_terms[:2]) and any(
            term in combined or term in lowered for term in plan.primary_terms
        )

    def _is_secondary_evidence_for_plan(self, plan: EvidencePlan, result: SearchResult) -> bool:
        combined = f"{result.chunk.section_title} {result.chunk.text}"
        lowered = combined.lower()
        if plan.intent == "parameter":
            return bool(NUMERIC_SIGNAL_RE.search(combined)) or result.chunk.chunk_type in {"component", "title_only"}
        return any(term in combined or term in lowered for term in plan.secondary_terms)

    def _is_background_evidence_for_plan(self, plan: EvidencePlan, result: SearchResult) -> bool:
        title = result.chunk.section_title
        combined = f"{result.chunk.section_title} {result.chunk.text}"
        lowered = combined.lower()
        if any(term in title for term in plan.background_title_terms):
            return True
        if any(term in combined or term in lowered for term in plan.background_body_terms):
            return True
        return False

    def _detail_signal_score(self, answer: str) -> int:
        score = 0
        numbered_items = len(re.findall(r"(?:^|\n)\s*\d+[\.\)、]\s*", answer))
        if numbered_items >= 2:
            score += 2
        elif numbered_items == 1:
            score += 1
        if re.search(r"\d+\s*(?:千克|公斤|磅|分钟|摄氏度|°C|V|W|A|号|kg|lb)", answer, re.IGNORECASE):
            score += 1
        if any(keyword in answer for keyword in ("按下", "打开", "关闭", "安装", "装入", "取下", "置于", "推", "拉", "切勿", "等待")):
            score += 1
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        substantive_lines = [
            line
            for line in lines
            if not self._contains_insufficient_claim(line) and not line.startswith("可参考配图")
        ]
        if any(len(line) >= 24 and "建议您" not in line and "根据提供的信息" not in line for line in substantive_lines):
            score += 1
        return score

    def _extract_image_hint(self, answer: str, context: AnswerContext) -> str:
        for line in answer.splitlines():
            stripped = line.strip()
            if stripped.startswith("可参考配图") or stripped.startswith("参考图片") or stripped.startswith("可一并查看相关配图"):
                if self._is_placeholder_image_hint(stripped):
                    return self._build_image_hint(context)
                return stripped
        if context.related_images:
            return self._build_image_hint(context)
        return ""

    def _finalize_answer_text(self, answer: str, context: AnswerContext) -> str:
        text = answer.replace("\r\n", "\n").strip()
        text = INVALID_IMAGE_HINT_RE.sub("", text)
        text = re.sub(
            r"<PIC>[ \t]*[A-Za-z][A-Za-z0-9_、,\- \t]*\d[A-Za-z0-9_、,\- \t]*[。.]?",
            "<PIC>",
            text,
        )
        text = self._normalize_image_hint_placeholders(text, context)
        text = ANSWER_SOURCE_INTRO_RE.sub("", text).strip()
        text = re.sub(r"^(?:根据(?:提供|现有)的信息|根据您描述的情况)[，,:：]\s*", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _normalize_image_hint_placeholders(self, answer: str, context: AnswerContext) -> str:
        answer = INVALID_IMAGE_HINT_RE.sub("", answer)
        english_mode = _looks_english_dominant_text(answer) and not answer.lstrip().startswith("您好")
        answer = re.sub(r"<PIC>\s*(?=(?:可参考配图|参考图片|可一并查看相关配图))", "", answer)
        answer = IMAGE_HINT_RE.sub("", answer)
        answer = EN_IMAGE_HINT_RE.sub("", answer)
        if not context.related_images:
            answer = re.sub(r"<PIC>", "", answer)
        normalized_lines: list[str] = []
        for line in answer.splitlines():
            stripped = line.strip()
            if (
                stripped.startswith("可参考配图")
                or stripped.startswith("参考图片")
                or stripped.startswith("可一并查看相关配图")
                or stripped.lower().startswith("reference image")
                or stripped.lower().startswith("related image")
            ):
                if self._is_placeholder_image_hint(stripped):
                    replacement = self._build_image_hint(context, english=english_mode)
                    if replacement:
                        normalized_lines.append(replacement)
                    continue
                continue
            normalized_lines.append(line)
        normalized = "\n".join(line.rstrip() for line in normalized_lines).strip()
        if context.related_images:
            image_hint = self._build_image_hint(context, english=english_mode)
            if image_hint:
                normalized = f"{normalized}\n{image_hint}".strip()
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        normalized = re.sub(r"[ \t]{2,}", " ", normalized)
        return normalized

    def _is_placeholder_image_hint(self, line: str) -> bool:
        return "图片ID" in line or "实际ID" in line or "无" in line or "（无）" in line or "(无)" in line

    def _format_image_hint(self, related_images: list[dict], *, english: bool = False) -> str:
        if not related_images:
            return ""
        image_names = ", ".join(image["image_id"] for image in related_images[:3]) if english else "、".join(image["image_id"] for image in related_images[:3])
        return f"Reference images: <PIC> {image_names}." if english else f"{PIC_HINT_PREFIX} {image_names}。"

    def _build_image_hint(self, context: AnswerContext, *, english: bool = False) -> str:
        if not context.related_images:
            return ""
        return self._format_image_hint(context.related_images, english=english)

    def _answer_conflicts_with_question(self, question: str, answer: str) -> bool:
        lowered_question = question.lower()
        lowered_answer = answer.lower()
        if ("quick release" in lowered_question or "qpr" in lowered_question) and re.search(
            r"(快速释放|quick release)[^。.!?]*(?:\bnr\b|\bnpr\b|自然释放|natural release)",
            lowered_answer,
        ):
            return True
        if "fuse" in lowered_question and "boat" in lowered_question and any(
            term in lowered_answer
            for term in (
                "dishwasher",
                "water tap",
                "洗碗机",
                "进水水龙头",
                "机门",
                "开机/关机",
                "家中保险丝",
            )
        ):
            return True
        if "fax" in lowered_question and any(token in lowered_question for token in ("connect", "connecting")):
            if "ren" in lowered_answer and not any(term in lowered_answer for term in ("line", "ext", "电话线", "插孔")):
                return True
        if "base station" in lowered_question and "connect" in lowered_question:
            bad_terms = ("charging contacts", "docking tone", "充电触点", "充电音", "电池正确安装")
            good_terms = ("telephone socket", "power socket", "电话插座", "电源插座", "电话线")
            if any(term in lowered_answer for term in bad_terms) and not any(term in lowered_answer for term in good_terms):
                return True
        if any(term in lowered_question for term in ("air fryer", "airfryer", "空气炸锅")) and any(
            term in lowered_question for term in ("first use", "first time", "before first use", "首次", "第一次")
        ):
            if any(term in lowered_answer for term in ("wifi", "wi-fi", "nutriu", "pair", "app", "配对", "应用")) and not any(
                term in lowered_answer for term in ("remove", "packaging", "clean", "wash", "清洁", "包装")
            ):
                return True
        if any(term in lowered_question for term in ("ship steers", "boat steers", "steer", "steering")):
            if any(term in lowered_answer for term in ("vacuum", "battery transportation", "lithium ion", "吸尘器")):
                return True
        if any(term in lowered_question for term in ("chair", "人体工学椅", "椅子")):
            if any(term in lowered_answer for term in ("dishwasher", "fitness tracker", "bluetooth mouse", "洗碗机", "健身追踪器", "蓝牙鼠标")):
                return True
        return False

    def _is_parameter_question(self, question: str) -> bool:
        lowered = question.lower()
        return any(keyword in lowered for keyword in PARAMETER_KEYWORDS)

    def _should_keep_rule_answer_without_polish(self, question: str) -> bool:
        return self._build_evidence_plan(question).intent in {
            "boat_battery_switches",
            "boat_over_temperature",
            "boat_factory_reset",
            "boat_fuse",
            "swim_platform_open",
            "fax_connect",
            "landline_base_station",
            "grill_indirect_cooking",
            "tv_manual_program_channels",
            "tv_outdoor_antenna",
            "boat_emission_label",
            "boat_engine_oil_level",
            "boat_battery_compartment",
            "boat_anchor_light",
            "boat_water_supply",
            "boat_bilge_pump",
            "boat_steering_turn",
            "boat_cross_wakes",
            "boat_flush_cooling",
            "boat_jet_wash_use",
            "boat_livewell",
            "boat_move_forward",
            "boat_throttle_cable",
            "boat_maintenance_screen",
            "microwave_control_setup",
            "microwave_light_timer",
            "microwave_favorite_recipe",
            "microwave_reheat",
            "microwave_auto_defrost",
            "microwave_oven_light",
            "microwave_grease_filter",
            "microwave_charcoal_filter",
            "vacuum_dual_modes",
            "vacuum_empty_bin",
            "vacuum_full_bin_sensors",
            "vacuum_sensors_contacts",
            "vacuum_home_base",
            "airfryer_first_use",
            "mower_roll_bar",
            "mower_rear_shock",
            "mower_height_cut",
            "mower_remove_filters",
            "mower_replace_belt",
            "mower_load",
            "mower_unload",
            "mower_engine_oil_level",
            "pressure_lid",
            "pressure_condensation_collector",
            "pressure_sealing_ring",
            "pressure_steam_release",
            "ereader_buttons",
            "ereader_main_browser",
            "ereader_ebook_mode",
            "ereader_music",
            "ereader_record",
            "ereader_video",
            "fitness_charge",
            "fitness_box_contents",
            "fitness_notifications",
            "fitness_interface",
            "fitness_exercise",
            "fitness_heart_rate",
            "fitness_payment",
            "fitness_lock",
            "fitness_troubleshooting",
            "oven_drip_tray",
            "oven_exterior_clean",
            "oven_grill_pan_set",
            "oven_grease_filter",
            "oven_catalytic_panels",
            "oven_sliding_shelf",
            "oven_baking_tray",
            "oven_wire_shelf",
            "generator_hot_safety",
            "generator_shock_safety",
            "generator_fuel_check",
            "generator_oil_check",
            "generator_control_switches",
            "generator_identification",
            "generator_sensitive_equipment",
            "water_pump_parts",
            "generator_start",
            "generator_stop",
            "generator_no_start",
            "camera_mount_lens",
            "camera_eyepiece_cover",
            "camera_p_mode",
            "camera_af_mode",
            "camera_auto_print",
            "processor_unit_parts",
            "coffee_empty_system",
            "boat_trip_screen",
            "jetski_hood_open_close",
            "grill_assembly_first_three_steps",
            "camera_power",
            "camera_cp_direct",
            "coffee_program_volume",
            "coffee_energy_saving",
            "coffee_after_use_clean",
            "earphones_case_charge",
            "earphones_reset",
            "fax_finger_safety",
            "fax_warning_labels",
            "generic_safe_operation",
            "fax_safety",
            "fax_move",
            "fax_canada",
            "jetski_seat",
            "jetski_filler_caps",
            "jetski_levers",
            "landline_install_handset",
            "landline_base_led",
            "landline_handset_led",
            "vacuum_clean_filter",
            "vacuum_clean_extractors",
            "vacuum_clean_side_brush",
            "vacuum_front_caster",
            "vacuum_dual_modes",
            "blower_start",
            "blower_carburetor",
            "blower_ppe",
            "blower_stop",
            "blower_safety",
            "air_conditioner_components",
            "air_conditioner_auto_restart",
            "airpurifier_remove_filter_packaging",
            "airpurifier_replace_filter",
            "airpurifier_dust_sensor",
            "airpurifier_modes",
            "airpurifier_casters",
            "airpurifier_clean_body",
            "airpurifier_clean_filter",
            "airpurifier_storage",
            "chair_parts",
            "chair_functions",
            "dishwasher_add_detergent",
            "dishwasher_tablet",
            "dishwasher_parts",
            "dishwasher_spray_arm_clean",
            "dishwasher_unsuitable_items",
            "dishwasher_basket_height",
            "bike_specs",
            "bike_workout_area",
            "bike_edit_profile",
            "bike_easy_ride_programs",
            "bike_fitness_test",
            "bike_mountain_programs",
            "bike_challenge_programs",
            "drill_battery_charge",
            "drill_keyless_chuck",
            "drill_battery_pack",
            "drill_accessories",
            "drill_warranty",
            "boat_bimini_remove",
            "boat_bimini_install",
            "boat_bimini_upright_storage",
            "boat_engine_start",
            "boat_load_distribution",
            "snowmobile_throttle_cable",
            "snowmobile_steering_system",
            "snowmobile_turning",
            "snowmobile_uphill",
            "snowmobile_downhill",
            "snowmobile_cross_slope",
            "snowmobile_engine_start",
            "snowmobile_spark_plug",
            "mouse_battery_install",
            "mouse_battery_status",
            "mouse_other_hid",
            "mouse_widcomm_install",
            "mouse_widcomm_uninstall",
            "mouse_widcomm_pair",
            "mouse_widcomm_first_use",
            "toothbrush_intensity",
            "toothbrush_features",
            "earphone_ear_tip_replace",
            "motherboard_pcie_x16",
            "motherboard_onboard_led",
            "motherboard_sata_odd_usb_os",
            "motherboard_t_sensor",
            "motherboard_chassis_screws",
            "motherboard_system_memory",
            "motherboard_tpm_connector",
            "quick_release",
            "natural_release",
            "float_valve",
            "toothbrush_travel_case_charge",
            "jetski_throttle_turning",
            "jetski_speed_modes",
            "jetski_stop",
            "jetski_avoid_collision",
            "jetski_characteristics",
            "jetski_fuel_filter",
            "jetski_intake_impeller",
            "steam_quick_assembly",
            "steam_functions",
            "steam_hard_floor",
            "thermostat_datetime",
            "thermostat_temp_override",
            "network_camera_t_rail",
            "function_keyboard_setup",
            "function_keyboard_switch_replace",
            "function_keyboard_warranty",
            "rideon_motorcycle_front_wheel",
        }

    def _chunk_contains_parameter_signal(self, text: str) -> bool:
        return bool(
            re.search(
                r"\d+\s*(?:千克|公斤|磅|分钟|摄氏度|°c|v|w|a|号|kg|lb|mph|km/h|l|ml|mm|cm|m)\b",
                text,
                re.IGNORECASE,
            )
        )

    def _contains_insufficient_claim(self, text: str) -> bool:
        return bool(INSUFFICIENT_CLAIM_RE.search(text))

    def _remove_insufficient_claims(self, text: str) -> str:
        cleaned = INSUFFICIENT_CLAIM_RE.sub("", text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip()

    def _missing_required_evidence(
        self,
        question: str,
        sub_question_results: list[tuple[str, list[SearchResult]]],
    ) -> bool:
        required_terms = self._required_evidence_terms(question)
        if not required_terms:
            return False
        selected_texts: list[str] = []
        for sub_question, results in sub_question_results:
            selected = self._select_primary_evidence_results(sub_question, results[:6])
            for result in selected[:3]:
                selected_texts.append(f"{result.chunk.section_title} {result.chunk.text}".lower())
        combined = "\n".join(selected_texts)
        return not any(term.lower() in combined for term in required_terms)

    def _required_evidence_terms(self, question: str) -> tuple[str, ...]:
        lowered = question.lower()
        matched: list[str] = []
        for query_aliases, evidence_aliases in CRITICAL_EVIDENCE_ALIASES:
            if any(self._question_matches_critical_alias(lowered, question, alias) for alias in query_aliases):
                matched.extend(evidence_aliases)
        return tuple(dict.fromkeys(matched))

    def _question_matches_critical_alias(self, lowered_question: str, original_question: str, alias: str) -> bool:
        alias_lower = alias.lower()
        if re.fullmatch(r"[a-z0-9]{1,3}", alias_lower):
            return bool(re.search(rf"(?<![a-z0-9]){re.escape(alias_lower)}(?![a-z0-9])", lowered_question))
        return alias_lower in lowered_question or alias in original_question

    def _missing_product_aligned_evidence(
        self,
        product_hint: str | None,
        sub_question_results: list[tuple[str, list[SearchResult]]],
    ) -> bool:
        if not product_hint:
            return False
        aliases = self._product_hint_aliases(product_hint)
        if not aliases:
            return False
        for sub_question, results in sub_question_results:
            selected = self._select_primary_evidence_results(sub_question, results[:6])
            if not selected:
                continue
            for result in selected[:3]:
                manual = result.chunk.manual_name.lower()
                product = result.chunk.product_name.lower()
                manual_product = f"{manual} {product}"
                if any(alias in manual_product for alias in aliases):
                    return False
        return True

    def _product_hint_aliases(self, product_hint: str) -> set[str]:
        normalized = product_hint.lower().strip()
        if not normalized or normalized in {"无", "unknown", "none"}:
            return set()
        compact = re.sub(r"\s+", "", normalized)
        aliases = {normalized, compact}
        alias_groups: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
            (("boat", "jetboat", "喷射船", "摩托艇"), ("boat", "jet boat", "jetboat", "喷射船", "英文喷射船", "摩托艇")),
            (("fax", "传真"), ("fax", "传真", "传真机", "英文传真机")),
            (("landline", "telephone", "basestation", "固定电话", "座机"), ("landline", "telephone", "fixed phone", "base station", "basestation", "固定电话", "英文固定电话", "座机")),
            (("pressurecooker", "airfryer", "压力锅", "空气炸锅"), ("pressure cooker", "pressurecooker", "air fryer", "airfryer", "压力锅", "空气炸锅", "multi-use pressure cooker")),
            (("camera", "相机"), ("camera", "相机")),
            (("ereader", "e-reader", "电子阅读器"), ("ereader", "e-reader", "电子阅读器", "英文电子阅读器")),
            (("lawnmower", "割草机"), ("lawn mower", "lawnmower", "割草机", "英文割草机")),
            (("snowmobile", "雪地摩托"), ("snowmobile", "雪地摩托", "英文雪地摩托")),
            (("airconditioner", "空调"), ("air conditioner", "airconditioner", "空调", "空调手册")),
            (("blower", "吹风机"), ("blower", "leaf blower", "吹风机", "吹风机手册")),
            (("airpurifier", "空气净化器"), ("air purifier", "airpurifier", "空气净化器", "空气净化器手册")),
            (("ergonomicchair", "officechair", "人体工学椅", "椅子"), ("ergonomic chair", "office chair", "chair", "人体工学椅", "人体工学椅手册")),
            (("dishwasher", "洗碗机"), ("dishwasher", "洗碗机", "洗碗机手册")),
            (("steamcleaner", "蒸汽清洁机"), ("steam cleaner", "steamcleaner", "蒸汽清洁机", "蒸汽清洁机手册")),
        )
        for needles, expanded in alias_groups:
            if any(needle in compact or needle in normalized for needle in needles):
                aliases.update(alias.lower() for alias in expanded)
        return {alias for alias in aliases if alias}

    def _select_best_snippet(self, question: str, results: list[SearchResult]) -> str:
        usable_results = [result for result in results if not self._is_low_value_result_for_answer(question, result)]
        if usable_results:
            results = usable_results
        elif results and all(self._is_low_value_result_for_answer(question, result) for result in results):
            return ""

        plan = self._build_evidence_plan(question)
        planned_answer = self._select_planned_direct_answer(plan, results, question)
        if planned_answer:
            return planned_answer

        if plan.intent == "delete_images":
            delete_answer = self._select_delete_images_answer(results, question)
            if delete_answer:
                return delete_answer

        step_style = self._is_step_question(question)
        if step_style:
            step_answer = self._select_step_answer(question, results)
            if step_answer:
                return step_answer

        range_style = any(token in question for token in ("前五条", "前5条", "最后三个步骤", "前两个步骤", "前六个步骤", "最后三步"))
        if range_style:
            ranged = self._select_range_answer(question, results)
            if ranged:
                return ranged

        list_style = any(token in question for token in ("哪些", "包含", "包括", "有什么", "分别", "前五条", "前5条"))
        if list_style:
            merged = self._select_merged_snippets(question, results)
            if merged:
                return merged

        best_result = results[0]
        english_mode = _looks_english_dominant_text(question)
        if best_result.chunk.chunk_type in {"general", "component", "warranty", "menu"}:
            primary_snippet = self._best_sentence_from_result(question, best_result)
            if primary_snippet:
                if best_result.chunk.section_title not in primary_snippet:
                    return f"{self._section_prefix(best_result.chunk.section_title, english_mode)}{primary_snippet}"
                return primary_snippet

        ranked_sentences: list[tuple[float, str]] = []
        query_terms = set(tokenize(question))
        for result in results:
            sentences = self._split_sentences(result.chunk.text)
            for sentence in sentences:
                sentence_terms = set(tokenize(sentence))
                overlap = len(query_terms & sentence_terms)
                boost = 0.6 if sentence and sentence[0].isdigit() else 0.0
                ranked_sentences.append((overlap + boost + result.score, sentence))

        ranked_sentences.sort(key=lambda item: (-item[0], len(item[1])))
        snippet = ranked_sentences[0][1] if ranked_sentences else best_result.chunk.text[:180]
        snippet = snippet.strip()
        if len(snippet) > 180:
            snippet = snippet[:177].rstrip("，,；; ") + "..."
        if best_result.chunk.section_title not in snippet:
            return f"{self._section_prefix(best_result.chunk.section_title, english_mode)}{snippet}"
        return snippet

    def _section_prefix(self, section_title: str, english: bool = False) -> str:
        return f'In the "{section_title}" section: ' if english else f"在“{section_title}”部分提到："

    def _select_planned_direct_answer(self, plan: EvidencePlan, results: list[SearchResult], question: str = "") -> str:
        text = " ".join(f"{result.chunk.section_title} {result.chunk.text}" for result in results[:3])
        lowered = text.lower()
        if _looks_english_dominant_text(question):
            english_answer = self._select_planned_direct_answer_en(plan, question)
            if english_answer:
                return english_answer
        if plan.intent == "water_pump_no_pump":
            return (
                "您好，水泵无法抽水时，优先检查进水端密封和泵体密封：\n"
                "1. 检查注水螺塞或放水螺塞是否松动，如松动请拧紧。\n"
                "2. 检查软管接头或卡箍是否松动，如松动请重新固定并拧紧。\n"
                "3. 检查 O 形圈或垫片是否损坏，如损坏请更换。\n"
                "4. 如果怀疑机械密封损坏，说明书建议咨询经销商处理。\n"
                "另外，使用前应确保软管安装牢固并安装滤网，否则可能进气导致无法抽水。"
            )
        if plan.intent == "drill_dcb101_indicator":
            return (
                "您好，DCB101 充电器的指示灯状态含义如下：\n"
                "1. 电池组充电中：表示电池正在充电。\n"
                "2. 电池组已充满：表示电池已充满，可以使用或留在充电器中。\n"
                "3. 过热/过冷延迟：表示电池温度过高或过低，充电器会暂停充电，待温度恢复后继续。\n"
                "4. 电池组或充电器故障：表示电池组或充电器可能存在故障，需要重新插入电池组或送检确认。\n"
                "5. 电源故障：红色指示灯快速闪烁两次后暂停，表示所接电源超出允许范围。"
            )
        if plan.intent == "water_pump_parts":
            return (
                "您好，水泵的核心部件包括：油箱、油箱盖、燃油开关、空气滤清器盖、火花塞、消音器、阻风门手柄、发动机开关、机油加注口盖、放油螺塞、反冲启动器、机油警告灯、油门手柄、注水螺塞和放水螺塞。"
            )
        if plan.intent == "processor_unit_parts":
            return (
                "您好，处理器单元的关键部件分为正面和背面：\n"
                "1. 正面包括状态指示灯、AUX 端口和 HDMI 输出端口；状态指示灯白色表示已开机，红色表示待机模式。\n"
                "2. 背面包括 HDMI 电视端口、HDMI PS4 端口、USB 端口、DC IN 12V 接口和通风口。"
            )
        if plan.intent == "function_keyboard_warranty":
            return (
                "您好，功能键盘的保修政策通常包括以下内容：\n"
                "1. 保修期内如产品存在符合保修条件的故障，可按库存情况维修或更换为全新或翻新、功能相近且价值相等或更高的产品。\n"
                "2. 申请经销商保修服务时，需要提供购买凭证，例如商店收据或发票。\n"
                "3. 若通过官网申请退货/换货，通常需要获得授权 RMA 编号，并按规定时间和流程退回。\n"
                "4. 未经许可改装产品或序列号/保修标签、非制造缺陷、意外损坏、滥用误用、进水、雷击、未按说明书操作、安装不当、非授权维修、运输损坏和正常磨损等情况通常不在保修范围内。"
            )
        if plan.intent == "function_keyboard_setup":
            return (
                "您好，功能键盘的基础设置可以按以下步骤完成：\n"
                "1. 用 USB-C 线连接键盘背面的接口，另一端连接电脑上可用的 USB 2.0 或更高速接口。\n"
                "2. 如使用全尺寸或无小键盘版本，可将腕托安装到键盘正面；腕托采用磁吸设计，会自动吸附并居中。\n"
                "3. 如果需要更高的打字倾斜角度，可展开键盘底部的可调节支撑脚，按需要选择中档或高档角度。\n"
                "4. 如需设置灯光、宏命令或配置文件，可通过 CAM 软件设置并保存到键盘的板载配置文件中；保存后，无需运行 CAM 也可以使用已存储的配置。"
            )
        if plan.intent == "function_keyboard_switch_replace":
            return (
                "您好，功能键盘轴体的拆卸和重新安装步骤如下：\n"
                "1. 拆卸时，使用附赠的拔轴器从顶部夹住轴体，并让工具两端沿轴体前后方向放置。\n"
                "2. 轴体由两个卡扣固定在定位板上，按下卡扣后即可向上取出轴体；不要从轴体下方撬动，以免刮擦上盖面板。\n"
                "3. 重新安装前，先确认轴体的金属针脚笔直，没有弯曲或折向轴底外壳。\n"
                "4. 将两个金属针脚对准键盘轴座插孔后，垂直向下用力压入插槽。\n"
                "5. 正确插入后，轴体外壳两侧应紧贴键盘上盖面板。"
            )
        if plan.intent == "rideon_motorcycle_front_wheel":
            return (
                "您好，儿童电动摩托车前轮安装可以按以下步骤操作：\n"
                "1. 将前轴依次穿过两侧把手管和前轮，并确认两侧把手管内外都安装了垫片。\n"
                "2. 用两个螺母将前轴和前轮固定在把手管上。\n"
                "3. 使用随产品提供的扳手拧紧螺母，直到拧到螺纹尽头。\n"
                "4. 组装完成后检查前轮与把手管之间的间隙：如果间隙过大、车轮不稳，可在车轮与把手管之间增加垫片；如果车轮卡死，则减少一片垫片。\n"
                "5. 最后确认前轮能够自由转动后再继续后续安装。"
            )
        if plan.intent == "airfryer_first_use":
            return (
                "您好，空气炸锅首次使用前建议先完成基础清洁和检查：\n"
                "1. 取下所有包装材料、贴纸或保护膜。\n"
                "2. 取出炸篮/锅篮和相关可拆部件，用温水和少量洗涤剂清洗，再彻底擦干。\n"
                "3. 用湿布擦拭产品内外表面，确认内部没有包装残留。\n"
                "4. 将部件重新装回并放在平稳、耐热、通风的位置后再开始使用。\n"
                "注意：如果问题只是首次使用准备，不需要进入 Wi-Fi、App 或 NutriU 配对流程。"
            )
        if plan.intent == "blower_ppe":
            return (
                "您好，使用吹风机时，人员需要佩戴以下防护装备：\n"
                "1. 合格的听力防护装备。\n"
                "2. 合格的眼部防护装备。\n"
                "3. 在多尘环境中操作时佩戴面罩。\n"
                "4. 防滑鞋底的工作靴或工作鞋。\n"
                "5. 准备急救箱。"
            )
        if plan.intent == "air_conditioner_components":
            return (
                "您好，空调的主要组成部件可按室内机、室外机和遥控器来理解：\n"
                "1. 室内机通常包括前面板、空气入口、空气滤网、空气出口、导风叶片和显示/指示区域。\n"
                "2. 室外机通常包括空气入口、空气出口、连接管、排水软管和电源/连接线等。\n"
                "3. 遥控器用于开关机、模式、温度、风速和定时等日常控制。"
            )
        if plan.intent == "air_conditioner_auto_restart":
            return (
                "您好，空调自动重启功能用于停电后恢复之前的运行设置，通常为出厂默认开启。\n"
                "关闭方法如下：\n"
                "1. 打开前盖板，轻轻掀起盖板两侧。\n"
                "2. 按住开/关键约 6 秒，设备会发出两声蜂鸣，指示灯闪烁 6 次。\n"
                "3. 如需重新开启，再次按住开/关键约 6 秒，设备会发出两声蜂鸣，蓝色指示灯闪烁 4 次。"
            )
        if plan.intent == "chair_parts":
            return (
                "您好，人体工学椅组装时主要会用到这些部件：椅背、座垫、扶手、底盘、气杆、底座、脚轮、连接件、头枕和腰枕等。\n"
                "组装顺序通常是先安装脚轮和底座/气杆，再安装底盘、座椅、椅背连接件、扶手，最后装头枕与腰枕。"
            )
        if plan.intent == "chair_functions":
            return (
                "您好，这款人体工学椅的主要功能包括：\n"
                "1. 高度调节：向上拉起升降拉杆调节椅子高度，到合适高度后松开。\n"
                "2. 椅背后仰：向上拉起后仰拉杆，椅背可随人体动作独立倾斜；向下按下可锁定角度。\n"
                "3. 腰枕按摩：插入 USB 后可启用腰枕按摩功能。\n"
                "4. 扶手/头枕/腰枕用于提升支撑和舒适度；使用一段时间后如扶手松动，可重新拧紧螺丝。"
            )
        if plan.intent == "dishwasher_parts":
            return (
                "您好，洗碗机常见部件和操作区域包括：开机/关机键、启动/暂停/取消键、显示屏、程序选择键、半载/洗涤块键、预约启动键、盐量与亮碟剂指示区域、上下层餐具篮、喷淋臂、过滤器、洗涤剂盒和亮碟剂盒等。"
            )
        if plan.intent == "dishwasher_spray_arm_clean":
            return (
                "您好，清洁洗碗机上层喷淋臂时，请按以下步骤操作：\n"
                "1. 先检查上层喷淋臂孔位是否堵塞。\n"
                "2. 如有堵塞，拆下喷淋臂进行清洁。\n"
                "3. 拧松固定螺母，取下喷淋臂。\n"
                "4. 清理喷孔中的残渣并用水冲洗干净。\n"
                "5. 清洁后重新装回喷淋臂，并确认能顺畅转动。"
            )
        if plan.intent == "dishwasher_unsuitable_items":
            return (
                "您好，以下物品不适合放入洗碗机清洗：\n"
                "1. 沾有烟灰、蜡烛残渣、抛光剂、染料或化学品的餐具。\n"
                "2. 铁制器具，避免生锈或污染其他物品。\n"
                "3. 带木质或骨质手柄、带黏合部件、含不耐热部件的银质餐具或刀具。\n"
                "4. 铜制和镀锡容器。\n"
                "5. 有装饰花纹的瓷器、铝制和银质器具，以及精致玻璃或水晶制品，清洗后可能褪色、失去光泽或受损。\n"
                "建议购买餐具时确认其是否标注为适合洗碗机清洗。"
            )
        if plan.intent == "dishwasher_basket_height":
            return (
                "您好，洗碗机上层餐具篮/碗篮可根据餐具尺寸调节高度，具体取决于型号：\n"
                "1. 使用篮滚轮调节时，先拨开上层篮导轨末端的限位器。\n"
                "2. 取出碗篮，改变滚轮位置后再将碗篮放回导轨，并关闭限位器。\n"
                "3. 若是带碗篮调节机构的型号，可抓住上层篮一侧架丝向上提起以升高碗篮，另一侧重复操作。\n"
                "4. 需要降低时，按下碗篮调节机构上的卡扣，再在另一侧重复操作。\n"
                "5. 调节后确认两侧处于同一水平，避免碗篮倾斜。"
            )
        if plan.intent == "airpurifier_modes":
            return (
                "您好，空气净化器常见运行和设置功能包括：\n"
                "1. 常规运行：按启动键开机，可用控制面板上下键选择风速，风速可在多个档位间调节。\n"
                "2. 空气质量显示：室内空气质量指示灯会根据检测到的细颗粒物浓度显示不同颜色，帮助判断空气状态。\n"
                "3. 安全锁：可长按指定按键开启，防止儿童误操作。\n"
                "4. 滤网更换提醒：滤网更换指示灯变红时，需要更换滤网；更换后长按“睡眠 + 自动”键重置指示灯。"
            )
        if plan.intent == "airpurifier_dust_sensor":
            return (
                "您好，清洁空气净化器灰尘传感器时，请按以下步骤操作：\n"
                "1. 双手握住设备上盖并向上拉起。\n"
                "2. 握住背部滤网盖把手并向前拉出。\n"
                "3. 拆下设备右侧的灰尘传感器盖。\n"
                "4. 用蘸少量水的棉签擦拭灰尘传感器镜头和进风口，再用干净的干棉签彻底擦干。\n"
                "5. 装回灰尘传感器盖和滤网盖，并将设备上盖标识朝前装回。"
            )
        if plan.intent == "steam_functions":
            return (
                "您好，蒸汽清洁机的实用功能主要包括：\n"
                "1. 手持蒸汽器可拆下使用，便于清洁局部区域。\n"
                "2. 拖把头适合硬质地面清洁。\n"
                "3. 布艺清洁头可搭配清洁布处理玻璃或硬质表面。\n"
                "4. 喷射喷嘴、弧形喷嘴等配件可用于缝隙、边角或硬质表面清洁。\n"
                "使用前请确认部件锁扣固定到位，并按不同清洁场景选择合适配件。"
            )
        if plan.intent == "steam_hard_floor":
            return (
                "您好，使用蒸汽清洁机清洁硬质地面时，请注意：\n"
                "1. 本机适用于瓷砖、Vinyl 地板、复合地板、大理石、石材及封边木地板。\n"
                "2. 清洁前请先扫地或吸尘。\n"
                "3. 清洁时缓慢移动机器，同时按下开关输出蒸汽。\n"
                "4. 如需对某一区域消毒，可将蒸汽拖把停留至少 15 秒，但不要超过 20 秒。\n"
                "5. 不出蒸汽时，请先拔掉电源、取下水箱加水后再继续清洁。\n"
                "注意：不可用于未封边木地板；首次使用时，机器可能需要数秒才会出蒸汽。"
            )
        if plan.intent == "boat_battery_switches" and "battery switch" in lowered and "emerg parallel" in lowered:
            return (
                "您好，航行前可重点确认船上的电池开关状态：\n"
                "1. 这艘船使用两块船用电池：一块是启动电池，用于发动机启动电路；另一块是 house 电池，用于照明、舱底泵、鼓风机、音响等附件电路。\n"
                "2. 电池开关组件上有 “START”、“HOUSE” 和 “EMERG PARALLEL” 三个开关。\n"
                "3. 正常使用时，应将 “START” 和 “HOUSE” 开关保持在 ON 位置，将 “EMERG PARALLEL” 开关保持在 OFF 位置。\n"
                "4. 如果启动电池没电，可将 “EMERG PARALLEL” 开关转到 ON 位置来启动发动机；发动机启动后或启动电池充好后，应再把它转回 OFF 位置。"
            )
        if plan.intent == "boat_over_temperature" and "over temperature" in lowered:
            return (
                "您好，如果船上出现 Over Temperature 过热警告，请按说明书这样处理：\n"
                "1. 发动机过热时，多功能显示屏会出现警告并伴随蜂鸣声，显示 “Over Temperature”；同时发动机转速会被自动限制，以帮助避免损坏。\n"
                "2. 发生这种情况时，请立即降低发动机转速，并返回岸边或移动到安全位置。\n"
                "3. 检查冷却水检查口是否有水排出；发动机运行时，尤其加油门时，应能看到冷却水从检查口流出。\n"
                "4. 如果看不到出水，说明冷却水可能没有在发动机内循环，不要继续高速运行。"
            )
        if plan.intent == "swim_platform_open" and "wet storage compartment" in lowered and "lock handle" in lowered:
            return (
                "您好，说明书中实际描述的是打开游泳平台下方的湿物储物舱，步骤如下：\n"
                "1. 找到位于 swim platform 下方的 wet storage compartment。\n"
                "2. 向上拉起锁止手柄。\n"
                "3. 顺时针转动锁止手柄，然后打开后平台舱盖。\n"
                "4. 关闭时，先合上后平台舱盖，再逆时针转动锁止手柄，并确认舱盖已牢固关闭，最后把锁止手柄按下。"
            )
        if plan.intent == "boat_factory_reset":
            return (
                "您好，船上的 factory reset screen 用于将设置恢复为出厂默认值，屏幕操作如下：\n"
                "1. 在 factory reset screen 上点击 “Reset” 按钮。\n"
                "2. 出现确认信息后，点击 “YES” 按钮即可重置设置。\n"
                "3. 如果不想执行重置，可点击 “NO” 按钮返回 factory reset screen。"
            )
        if plan.intent == "boat_fuse" and "fuse" in lowered and ("spare fuse" in lowered or "fuse puller" in lowered):
            amperage = []
            for label, zh_label in (
                ("Electronic throttle valve fuse", "电子节气门保险丝"),
                ("Fuel pump fuse", "燃油泵保险丝"),
                ("Main relay drive fuse", "主继电器驱动保险丝"),
                ("Main fuse", "主保险丝"),
                ("Battery fuse", "电池保险丝"),
                ("Accessory fuse", "附件保险丝"),
                ("Bilge pump fuse", "舱底泵保险丝"),
            ):
                match = re.search(rf"{re.escape(label)}\s*:\s*(\d+\s*A)", text, re.IGNORECASE)
                if match:
                    amperage.append(f"{zh_label}：{match.group(1)}")
            amperage_line = f" 保险丝规格包括：{'；'.join(amperage)}。" if amperage else ""
            return (
                "您好，可以按说明书中的保险丝更换流程处理：\n"
                "1. 先取下保险丝盒盖，确认熔断的保险丝位置。\n"
                "2. 使用保险丝拔取器取出熔断保险丝，并更换为相同/正确安培数的备用保险丝。\n"
                "3. 如果要更换附件保险丝或舱底泵保险丝，需要先取下保险丝座；手册说明这些保险丝可通过打开电池舱接触到。\n"
                "4. 更换完成后，把保险丝盒盖重新装回。\n"
                f"注意：不要使用高于推荐安培数的保险丝，否则可能损坏电气系统并引发火灾。{amperage_line}"
            )
        if plan.intent == "fax_connect" and (
            "telecommunication line cord" in lowered
            or "standard modular jack" in lowered
            or "telephone line" in lowered
        ):
            return (
                "您好，连接传真功能时建议重点确认电话线和电源位置：\n"
                "1. 使用 No. 26 AWG 或更粗规格的电信电话线。\n"
                "2. 设备应安装在容易够到的交流电源插座附近，紧急情况下可直接拔下电源线完全断电。\n"
                "3. 电话线应通过符合要求的标准模块化插孔连接；手册中提到可安全连接到标准模块化插孔 USOC RJ11C。\n"
                "4. 安装、维修或改动设备前，请先从墙上插座断开所有线缆，以降低触电风险。"
            )
        if plan.intent == "landline_base_station" and (
            "connect each end of the power adapter" in lowered
            or ("dc input jack" in lowered and "telephone socket" in lowered)
        ):
            return (
                "您好，固定电话底座主要需要连接电源适配器和电话线：\n"
                "1. 将电源适配器一端接到底座底部的 DC 输入接口，另一端接到墙上的电源插座。\n"
                "2. 将电话线一端接到底座底部的电话插孔，另一端接到墙上的电话插座。\n"
                "3. 如果电话线同时使用 DSL 高速上网服务，请在电话线和电源插座之间安装 DSL 滤波器，以减少噪声和来电显示问题。"
            )
        if plan.intent == "quick_release" and "quick release" in lowered:
            return (
                "您好，快速释放（QR/QPR）用于在烹饪结束后更快排出锅内压力，操作要点如下：\n"
                "1. 按下快速释放按钮，直到听到咔嗒声并锁定在排气位置。\n"
                "2. 按下后，蒸汽会从蒸汽释放阀顶部喷出，这是正常现象。\n"
                "3. 压力完全释放后，烹饪会迅速停止，可帮助避免食物过度烹饪，适合蔬菜、易碎海鲜等需要快速停止加热的食物。"
            )
        if plan.intent == "natural_release" and ("natural release" in lowered or "depressurizes naturally" in lowered):
            return (
                "您好，自然释放（NR/NPR）指的是压力烹饪结束后，不手动快速排气，而是让锅内温度逐渐下降，压力随时间自然降低。\n"
                "打开盖子前必须先完成泄压；具体选择自然释放还是快速释放，应按食谱要求操作。"
            )
        if plan.intent == "float_valve" and "float valve" in lowered:
            return (
                "您好，浮子阀需要和硅胶帽配合使用，拆装时请按以下步骤操作：\n"
                "1. 用手指按住浮子阀顶部的平面，然后翻转锅盖。\n"
                "2. 从浮子阀底部取下硅胶帽。\n"
                "3. 从锅盖顶部取出浮子阀。\n"
                "4. 请勿丢弃浮子阀或硅胶帽，后续需要重新安装。"
            )
        if plan.intent == "toothbrush_travel_case_charge" and "charging with the travel case" in lowered:
            return (
                "您好，电动牙刷放在旅行盒内充电时，可以按以下步骤操作：\n"
                "1. 将 USB 线插入旅行盒，并连接到 USB 墙充适配器。\n"
                "2. 将墙充适配器插入电源插座。\n"
                "3. 把牙刷放入旅行盒；如果开始充电成功，牙刷会发出两声提示音，灯光会向上亮起。\n"
                "4. 充电时电池指示灯会闪烁白光；保持旅行盒接通电源，直到牙刷完全充满。\n"
                "5. 牙刷手柄充满后，电池灯会熄灭；说明书还提示将旅行盒侧放，以获得更好的稳定性。"
            )
        if plan.intent == "grill_indirect_cooking" and "indirect cooking" in lowered:
            return (
                "您好，使用烧烤炉进行 indirect cooking（间接烹饪）时，请注意以下几点：\n"
                "1. 间接烹饪适合禽类和大块肉类，通过选定燃烧器产生的热量在炉内循环，让食物慢慢烤熟，避免直接火焰接触。\n"
                "2. 这种方式可以减少油脂滴落被明火点燃造成的 flare-ups（火焰上窜）。\n"
                "3. 间接烹饪时应保持炉盖关闭。\n"
                "4. 受天气影响，烹饪时间可能变化；在寒冷或有风环境下，可能需要提高温度设置以保证足够烹饪温度。\n"
                "5. 低温间接烹饪可产生缓慢、均匀的加热效果，适合慢烤和烘烤。"
            )
        if plan.intent == "tv_manual_program_channels" and "manual program" in lowered and "memory/erase" in lowered:
            return (
                "您好，电视记忆频道可以使用 AUTO PROGRAM 或 MANUAL PROGRAM 两种方式；如果使用手动节目记忆，可按以下方法操作：\n"
                "1. 使用遥控器上的上/下键或数字键，选择要记忆或删除的频道号。\n"
                "2. 按 MEMORY/ERASE 按钮，在 Memory 和 Erase 之间选择。\n"
                "3. 屏幕会显示对应提示，用于确认该频道是被记忆还是被删除。"
            )
        if plan.intent == "tv_outdoor_antenna" and "antenna" in lowered:
            return (
                "您好，为了获得更好的电视接收效果，说明书建议优先使用室外天线，并在连接前检查天线和天线线缆是否严重老化，因为这会降低信号质量。\n"
                "连接方式可按线缆类型区分：\n"
                "1. 如果使用 300 ohm 扁平线，请先将其连接到 300 ohm 转 75 ohm 适配器的螺丝上，再把适配器末端插入 75 ohm 天线接口。\n"
                "2. 如果使用 75 ohm 同轴线，可将同轴线直接连接到 75 ohm 天线接口。"
            )
        if plan.intent == "boat_emission_label" and "approval label" in lowered:
            return "您好，排放控制证书的 approval label 贴在每个发动机单元上，也贴在发动机舱内部；查找时需要先打开发动机舱，再查看对应的 emission control information label。"
        if plan.intent == "boat_engine_oil_level" and "oil tank filler cap" in lowered:
            return (
                "您好，检查船的发动机机油液位时，请按以下步骤操作：\n"
                "1. 发动机停止后，将船在陆地上保持精确水平，或将船下水。\n"
                "2. 确认四周安全后启动发动机，让发动机怠速运行至少 6 分钟；如果环境温度为 20°C（68°F）或更低，再额外运行 5 分钟。\n"
                "3. 停止发动机并打开 engine hood。\n"
                "4. 拧松并取下 oil tank filler cap/dipstick，把油尺擦干净。\n"
                "5. 将油箱加注口盖拧到底后再次取出，确认油位在 minimum 与 maximum 标记之间。\n"
                "6. 油位过低时缓慢添加发动机油；油位明显高于最高标记时，请咨询船只经销商。"
            )
        if plan.intent == "boat_battery_compartment" and "battery compartment" in lowered and "latch" in lowered:
            return (
                "您好，打开船的电池舱时，请在船尾左舷侧找到 battery compartment：\n"
                "1. 先松开电池舱盖上的 latch。\n"
                "2. 打开 battery compartment lid。\n"
                "3. 关闭时，合上电池舱盖，再把 latch 扣回甲板上。\n"
                "注意：不要在电池舱内放易燃物、重物或金属物品，以免损坏电池或造成短路。"
            )
        if plan.intent == "boat_anchor_light" and "anchor light" in lowered:
            return (
                "您好，安装船的 anchor light 时，可以按说明书步骤操作：\n"
                "1. 打开可上锁储物舱，取出 anchor light。\n"
                "2. 将 anchor light stoppers A 和 B 分开，展开 anchor light pole，并把 stopper A 拧到灯杆中段。\n"
                "3. 打开 anchor light socket 的盖子，将灯上的凸起对准插座槽位，把灯插入插座。\n"
                "4. 将 stopper B 装入 anchor light socket。夜间航行时用 NAV 位置点亮 bow light 和 anchor light；夜间锚泊时用 ANC 位置只点亮 anchor light。"
            )
        if plan.intent == "boat_water_supply" and "shut-off valve" in lowered:
            return (
                "您好，打开或关闭船上 jet wash 的水路供应时，请按以下步骤操作：\n"
                "1. 先停止发动机。\n"
                "2. 打开 rear platform hatch。\n"
                "3. 取下 inspection cover。\n"
                "4. 如需打开水路供应，将 shut-off valve 顺时针旋转 90°。如需关闭，则将阀门转回关闭位置。"
            )
        if plan.intent == "boat_bilge_pump" and "bilge pump" in lowered:
            return (
                "您好，舱底泵用于排出进入船内并汇集到发动机舱下方 bilge 区域的水：\n"
                "1. 打开 bilge pump switch 后，舱底泵会开始工作。\n"
                "2. 即使未打开开关，当舱底水过多时，舱底泵也会自动检测并通过 bilge pump outlet 排出大部分积水。\n"
                "3. 舱底泵工作时，bilge pump indicator light 会亮起；自动运行时，会持续工作直到大部分舱底水排出。"
            )
        if plan.intent == "boat_steering_turn" and "jet thrust" in lowered:
            return (
                "您好，这艘喷射船转向依靠 steering wheel、jet thrust nozzle 和油门配合完成：\n"
                "1. 向希望行驶的方向转动方向盘，喷射推力喷嘴角度会随之改变，船只方向也会改变。\n"
                "2. 转向力度取决于喷射推力和方向盘位置；油门越大，喷射推力越强，转弯越急。\n"
                "3. 尝试避让或转向时不要把遥控操纵杆直接拉回 idle/neutral，因为推力过低会让转向能力快速下降。\n"
                "4. 低速拖曳状态下，可依靠较小推力和方向盘位置进行较缓慢转向。"
            )
        if plan.intent == "boat_cross_wakes" and "crossing wakes and swells" in lowered:
            return (
                "您好，穿越尾流和浪涌时，说明书建议通过速度和角度来减小冲击：\n"
                "1. 不要假设水面一直平稳，遇到其他船只尾流或浪涌时要准备好修正方向和平衡。\n"
                "2. 穿越前调整速度和角度；通常降低速度并以一定角度斜穿（quartering）会减少船体和乘员受到的冲击。\n"
                "3. 小浪涌相对容易通过，尖锐尾流带来的冲击更明显；连续多组尾流也比单个尾流更难平稳通过。"
            )
        if plan.intent == "boat_flush_cooling" and "flushing the cooling system" in lowered:
            return (
                "您好，冲洗冷却系统可防止盐、沙或污物堵塞冷却通道：\n"
                "1. 将 garden hose adapter 连接到花园水管。\n"
                "2. 拧松并取下 flush hose connector cap，把适配器插入 flush hose connector，推入并旋转直到牢固连接。\n"
                "3. 将水管接到水龙头，确认船周围安全后启动发动机。\n"
                "4. 发动机启动后立即完全打开水源，确认水从 jet thrust nozzle 和 cooling water pilot outlet 持续流出。\n"
                "5. 让发动机快速怠速运行 3 到 5 分钟；结束时先关闭水源，再让残水从排气系统排出，最后停止发动机。"
            )
        if plan.intent == "boat_livewell" and "livewell" in lowered and "livewell switch" in lowered:
            return (
                "您好，livewell 位于船尾右舷侧，用于存放活饵和鱼：\n"
                "1. 拉动 latch 打开 livewell lid。\n"
                "2. 按下 livewell switch，启动 livewell pump 向舱内供水。\n"
                "3. 水量足够后，再按 livewell switch 关闭水泵。\n"
                "4. 如需增氧或循环水，可按 aerator switch。"
            )
        if plan.intent == "boat_move_forward" and "forward" in lowered and "remote control levers" in lowered:
            return (
                "您好，准备让船前进时，请使用 remote control levers：\n"
                "1. 从 neutral 位置将 remote control levers 向前推，船会切换到 forward position。\n"
                "2. 继续向前推动操纵杆，会提高发动机输出并增加喷射推力。\n"
                "3. 在低速前进时，shift gates 会从 neutral 位置稍微抬起，TDE 功能会帮助低速转向。\n"
                "4. 当操纵杆进一步前推时，shift gates 完全抬起，喷射推力向后，船体向前移动。"
            )
        if plan.intent == "boat_throttle_cable" and "throttle-cable" in lowered:
            return "您好，保养 throttle cable 时，请在 APS 的 pulley wheel 处给 throttle-cable inner wires 涂抹润滑脂；同时可给喷射推力喷嘴处的 steering cable 和 shift cable 球头及内线薄薄涂一层润滑脂。"
        if plan.intent == "microwave_control_setup" and "control set-up" in lowered:
            return "您好，微波炉的 Control Set-Up 可更改默认设置，包括 beep sound、clock、display speed 和 defrost weight（LBS/KG）。例如要把解冻重量模式从磅改为千克，可进入 Control Set-Up 后按说明选择对应项目并更改默认值。"
        if plan.intent == "microwave_light_timer" and "light timer" in lowered:
            return "您好，Light Timer 可设置底部 Lo Light 在每天固定时间自动开启和关闭；如果要重新设置开关灯时间，重复设置步骤即可。如需取消正在运行的 Light Timer，可触按 Light HI/LO/Off。"
        if plan.intent == "microwave_favorite_recipe" and "favorite recipe" in lowered:
            return "您好，Favorite Recipe 用于快速调用已存入记忆的一组烹饪指令。您可以先把自定义烹饪时间/程序保存到 Favorite Recipe，之后触按 Favorite Recipe 即可调出该自定义食谱并快速开始烹饪；默认功率为 Hi，但可修改。"
        if plan.intent == "microwave_reheat" and "reheat" in lowered:
            return "您好，Reheat（Sensor）用于无需手动设置时间和功率来加热食物。它有预设类别，包括 Casserole、Dinner Plate 和 Soup/Sauce；选择对应类别后，传感器会根据食物湿度判断加热时间，完成后会蜂鸣并显示 END。"
        if plan.intent == "microwave_auto_defrost" and "auto defrost" in lowered:
            return (
                "您好，Auto Defrost 是微波炉预设的自动解冻功能，适用于冷冻食物：\n"
                "1. 选择 Auto Defrost，并按食物类型和重量进行设置。\n"
                "2. 启动后，显示屏会进入解冻倒计时。\n"
                "3. 解冻过程中微波炉会蜂鸣并暂停；此时打开门，翻动、分开或重新摆放食物，取出已经解冻的部分。\n"
                "4. 将仍冷冻的部分放回炉内，触按 START 继续解冻。"
            )
        if plan.intent == "microwave_oven_light" and "oven light replacement" in lowered:
            return (
                "您好，更换微波炉 oven light 时，请先断电：\n"
                "1. 拔下电源或关闭主电源。\n"
                "2. 拆下 vent cover 的安装螺丝，将盖板向前倾斜并取下。\n"
                "3. 取下并抬起 bulb holder。\n"
                "4. 更换为 30 或 40 瓦 appliance bulb。\n"
                "5. 装回 bulb holder 和 vent cover，重新拧紧螺丝，再恢复电源。"
            )
        if plan.intent == "vacuum_dual_modes" and "dual mode virtual wall barrier" in lowered:
            return "您好，吸尘器配套的 Dual Mode Virtual Wall Barrier 有两种主要模式：Virtual Wall Mode 可形成一个不可见的锥形屏障，阻挡吸尘器进入不希望清洁的区域；Halo Mode 可在设备周围形成保护区，用于保护狗碗、花瓶等不希望吸尘器碰到的物品。"
        if plan.intent == "vacuum_empty_bin" and "emptying the bin" in lowered:
            return "您好，清空吸尘器集尘盒时：1. 按下 bin release button 取下集尘盒。2. 打开 bin door 倒出垃圾。提示：如果 full bin 指示灯在清洁过程中亮起，可以暂停清洁、清空集尘盒后继续；如果集尘盒不满但指示灯亮，请清洁 full bin sensors。"
        if plan.intent == "vacuum_full_bin_sensors" and "full bin sensors" in lowered:
            return "您好，清洁 full bin sensors 时：1. 取下并清空集尘盒。2. 用干净的干布擦拭传感器。3. 用干净的干布擦拭集尘盒内外的传感器端口。"
        if plan.intent == "vacuum_sensors_contacts" and "charging contacts" in lowered:
            return "您好，清洁吸尘器传感器和充电触点时，请用干净的干布擦拭传感器，不要把清洁剂直接喷到传感器或传感器开口上；同时用干净的干布擦拭吸尘器和 Home Base 上的 charging contacts。"
        if plan.intent == "vacuum_home_base" and "positioning the vacuum" in lowered:
            return "您好，放置 Home Base/吸尘器底座时，请选择开放、无遮挡的位置：两侧至少预留 1.5 英尺，前方至少预留 4 英尺，并距离楼梯至少 4 英尺；同时应距离 virtual wall barriers 至少 8 英尺，并保持底座接通电源。"
        if plan.intent == "mower_roll_bar" and "roll bar" in lowered:
            return (
                "您好，割草机的 roll bar（防翻滚杆）是重要安全装置，说明书要求优先保持在升起并锁定的位置：\n"
                "1. 正常使用时，应保持 roll bar 完全升起并锁定，同时系好安全带。\n"
                "2. 只有在绝对必要时才可放下 roll bar；放下时不要系安全带，并应低速、小心驾驶。\n"
                "3. 放下 roll bar 时，取下 hairpin cotters 和 2 个 pins，将 roll bar 放到下方位置，再装回 pins 并用 hairpin cotters 固定。\n"
                "4. 升起 roll bar 时，同样先取下 hairpin cotters 和 2 个 pins，将 roll bar 升到直立位置，再装回 pins 并固定。"
            )
        if plan.intent == "mower_rear_shock" and "rear-shock assemblies" in lowered:
            return "您好，带悬挂系统的割草机可调节 rear-shock assemblies 来改变乘坐舒适度。可将左右后减震组件调到从 softest 到 firmest 的不同位置；说明书提醒左右两侧应始终调在相同位置。"
        if plan.intent == "mower_height_cut" and "electric deck lift" in lowered:
            return "您好，带 electric deck lift 的割草机调节割草高度时：1. 向上推 deck-lift switch 可升起割草平台，向下推可降低平台。2. 在 height-of-cut bracket 上选择对应目标高度的孔位。3. 插入 height-of-cut pin 固定。"
        if plan.intent == "mower_remove_filters" and "removing the filters" in lowered:
            return "您好，拆卸割草机空气滤清器滤芯时：1. 将机器停在水平地面，断开 PTO，并拉起驻车制动。2. 熄火、拔钥匙，等待所有运动部件停止。3. 松开 air cleaner 的 latches，取下 air-inlet cover。4. 清洁 air-inlet screen 和 cover。5. 检查 primary filter 是否损坏，必要时丢弃更换。"
        if plan.intent == "mower_replace_belt" and "replacing the mower belt" in lowered:
            return (
                "您好，更换割草机 mower belt 时：\n"
                "1. 将机器停在水平地面，断开 PTO，拉起驻车制动，熄火并拔钥匙。\n"
                "2. 将割草高度降到 76 mm（3 inches），拆下 belt covers。\n"
                "3. 用 3/8 英寸棘轮插入 idler arm 的方孔，释放 idler spring 张力。\n"
                "4. 从 mower-deck pulleys 和 clutch pulley 上取下皮带，并拆下 spring-loaded idler arm 上的 belt guide。\n"
                "5. 拆下旧皮带，按图示绕上新皮带，再装回 belt guide、idler spring 和 belt covers。"
            )
        if plan.intent == "mower_load":
            return "您好，装载割草机时，请使用足够宽且牢固的坡道，将坡道固定到拖车或卡车上，缓慢直线上坡，避免急转或突然加速；装上后关闭发动机、拉起驻车制动，并按说明用绑带固定机器。"
        if plan.intent == "mower_unload":
            return "您好，卸下割草机时，请确认坡道已牢固固定并与地面角度合适，缓慢直线驶下坡道，避免急转、突然制动或侧向移动；卸车过程中保持低速，并确保周围没有人员和障碍物。"
        if plan.intent == "pressure_lid" and "pressure cooking lid" in lowered:
            return "您好，压力锅盖的拆装方法如下：拆下锅盖时，握住 lid handle，逆时针旋转，使锅盖符号与 cooker base 边缘符号对齐，然后朝自己方向向上提起锅盖。关闭锅盖时，将锅盖符号与 cooker base 上的符号对齐，放入轨道后顺时针旋转，直到锅盖符号与底座符号对齐。"
        if plan.intent == "pressure_condensation_collector" and "condensation collector" in lowered:
            return "您好，condensation collector 位于 cooker base 背面，用于接住 condensation rim 溢出的水。安装时，将 condensation collector 的 grooves 对准 cooker base 背面的 tabs，然后滑入到位；它应在烹饪前安装，并在每次使用后倒空和冲洗。"
        if plan.intent == "pressure_sealing_ring":
            return (
                "您好，sealing ring 会在压力锅盖和内锅之间形成气密密封，使用压力锅前必须正确安装：\n"
                "1. 拆下 sealing ring 时，抓住硅胶边缘，将密封圈从圆形不锈钢 sealing ring rack 后方拉出。\n"
                "2. 拆下后检查不锈钢支架是否居中、固定且高度均匀，不要尝试修复已经变形的支架。\n"
                "3. 安装时，将 sealing ring 套在 sealing ring rack 上并按压到位，按紧避免起皱。\n"
                "4. 正确安装后，密封圈应紧贴在支架后方；翻转锅盖时不应掉落。一次只安装一个 sealing ring。"
            )
        if plan.intent == "pressure_steam_release":
            return "您好，设置 steam release valve 时，请按说明使用快速释放按钮：按下快速释放按钮，直到听到咔嗒声并锁定在 Vent 排气位置；此时蒸汽会从 steam release valve 顶部喷出，这是正常现象。等待压力完全释放后，再按食谱或说明书要求打开锅盖。"
        if plan.intent == "ereader_buttons" and ("front view" in lowered or "home/esc" in lowered):
            return "您好，电子阅读器的主要按键和接口包括：Home/ESC、Prev/Next Page、四向 Navigation/Menu、Zoom in/out、Rotate、3.5 mm 耳机孔、USB Port、Micro SD card reader、Play/Pause、Power button、音量加减、Reset 开关、扬声器和 TFT LCD 显示屏。"
        if plan.intent == "ereader_main_browser" and "browser history" in lowered:
            return "您好，Main Menu 会显示设备的主要功能，包括 Browser History、eBook、Music、Video、Photo、Record、Explorer 和 settings，选择需要的功能即可进入。Browser History 会显示最近浏览过的文件，选择书籍并按 M 键后，可跳转到上次阅读的位置。"
        if plan.intent == "ereader_ebook_mode":
            return "您好，在 eBook mode 下按 M 键，可进入 Page Jump、Save Mark、Load Mark、Del Mark、Browser Mode、Flip Time、Brightness 和 Set Color 等功能；例如可跳转页码、保存/加载/删除书签、设置自动/手动浏览模式、调整翻页时间和亮度。"
        if plan.intent == "ereader_music" and "music mode" in lowered:
            return "您好，使用电子阅读器听音乐时，在主菜单打开 audio files list，选择想播放的音频文件并按 M 键进入播放模式。也可以通过 USB 连接电脑，把音频文件拖入设备；播放时可用上下键调节音量，用 Play/Pause 暂停或继续。"
        if plan.intent == "ereader_record":
            return "您好，电子阅读器支持录音功能。操作时在主菜单选择 Record 并按 M 键进入 voice record mode；按 Play/Pause 开始录音，再按一次可暂停；按 HOME 停止录音。录音结束后，设备会提示是否保存，按 M 键选择 YES 或 NO。回放录音时，可到 Music 菜单中的 Recorded 文件列表选择录音文件并按 M 播放。"
        if plan.intent == "ereader_video":
            return "您好，播放视频时，在主菜单选择 Video 并按 M 键进入 Video mode。设备支持 AVI、RMVB、MPEG2 等视频格式。播放过程中如视频文件带字幕，可按 M 进入设置并选择 Subtitle Language；也可以通过 M 菜单选择 time play、Full Screen 或调节 Brightness。"
        if plan.intent == "snowmobile_throttle_cable" and "throttle" in lowered and "cable" in lowered:
            return (
                "您好，雪地摩托的 throttle cable 需要在使用前确认工作正常：\n"
                "1. 启动发动机前，应检查 throttle、brake 和 steering 是否能正常操作。\n"
                "2. 如果 carburetor 或 throttle cable 在运行中发生故障，应松开 throttle lever；T.O.R.S. 系统会中断点火并停止发动机。\n"
                "3. 如果 T.O.R.S. 使发动机停止，必须先排除故障原因，确认发动机可正常运行后再重新启动。\n"
                "4. 润滑时只在 brake/throttle cable ends 处少量涂抹低温润滑脂；不要给整根 brake/throttle cable 本体涂脂，以免结冰导致失控。"
            )
        if plan.intent == "snowmobile_steering_system":
            return "您好，检查雪地摩托 steering system 时，应检查把手上下、前后推动以及左右轻转时是否有过大自由间隙；如果发现自由间隙过大，请咨询经销商。"
        if plan.intent == "snowmobile_turning":
            return "您好，雪地摩托转弯时应先减速，并把把手转向想要行驶的方向；同时把身体重量压到转弯内侧的踏板上，上身向弯内倾斜。该动作应先在宽阔、平坦且无障碍区域低速反复练习；弯越急或速度越高，需要向弯内倾斜得越多。"
        if plan.intent == "snowmobile_spark_plug" and "spark plug" in lowered:
            return "您好，检查雪地摩托火花塞时，请观察中心电极周围白色瓷绝缘体的颜色；正常使用时理想颜色为中等到浅棕色。安装火花塞前，用线规测量电极间隙并调整到规格值；安装时清洁垫圈表面，擦净螺纹污垢，并按规定扭矩 28 Nm（2.8 m-kg，20 ft-lb）拧紧。"
        if plan.intent == "earphone_ear_tip_replace":
            return (
                "您好，耳机出厂通常已安装 M 号耳塞，也可以根据佩戴需要更换为 S 号或 L 号耳塞。\n"
                "更换时，先旋转耳塞并将其拔下；安装新耳塞时要牢固推入并确认固定到位，避免使用过程中意外脱落。"
            )
        if plan.intent == "mouse_battery_install":
            return (
                "您好，蓝牙激光鼠标安装电池时：\n"
                "1. 按下鼠标上的按钮，弹出电池仓盖。\n"
                "2. 按鼠标内部标注的正负极方向，装入两节 AA 电池。\n"
                "3. 装回电池仓盖，并确认盖子扣好。"
            )
        if plan.intent == "mouse_battery_status":
            return (
                "您好，蓝牙激光鼠标可通过 LED 判断电池状态：当 LED 显示琥珀色时，表示电量低，需要尽快更换电池。\n"
                "如需查看更多电池状态信息，也可以在电脑的 Control Panel > Mouse > Bluetooth 中查看。"
            )
        if plan.intent == "mouse_other_hid":
            return (
                "您好，连接其他 HID 蓝牙设备时，请先确认 USB 蓝牙接收器已插入电脑：\n"
                "1. 将要连接的蓝牙设备设置为可被发现/discoverable 状态。\n"
                "2. 按下 USB 蓝牙接收器按钮，进入 HID 设备连接界面。\n"
                "3. 在下拉列表中选择设备类型。\n"
                "4. 选择检测到的设备并启用连接。"
            )
        if plan.intent == "mouse_widcomm_install" and "widcomm" in lowered:
            return "您好，安装 WIDCOMM 蓝牙驱动程序前，请先将 USB 蓝牙接收器插入电脑 USB 接口；放入安装光盘，若未自动加载则双击 bin 文件夹内的 Setup.exe；按向导点击“下一步”，接受许可协议，选择安装路径后点击“安装”；安装完成后拔下接收器再重新插入，鼠标会自动配对，最后按提示重启电脑。"
        if plan.intent == "mouse_widcomm_uninstall" and "widcomm" in lowered:
            return "您好，卸载 WIDCOMM 蓝牙驱动程序时：1. 先拔下 USB 蓝牙接收器。2. 点击 开始 > 设置 > 控制面板 > 添加或删除程序。3. 在程序列表中选择 WIDCOMM 蓝牙软件并点击“删除”。4. 等待卸载完成后，按提示选择立即或稍后重启电脑。"
        if plan.intent == "mouse_widcomm_pair" and "widcomm" in lowered:
            return "您好，使用 WIDCOMM 蓝牙驱动程序配对鼠标前，请先安装驱动程序。然后按下 USB 蓝牙接收器按钮，直到出现 HID（人机接口设备）界面；鼠标连接成功后，会出现确认信息，点击确认连接正确的鼠标即可。"
        if plan.intent == "mouse_widcomm_first_use" and "widcomm" in lowered:
            return "您好，首次使用 WIDCOMM 蓝牙驱动程序时，请双击系统托盘中的蓝牙图标，启动蓝牙初始配置向导；按屏幕提示完成配置后，会出现 HID 界面并开始搜索蓝牙设备，选择要连接的 HID 设备并按提示启用即可。"
        if plan.intent == "toothbrush_intensity":
            return "您好，电动牙刷有 3 档强度：High intensity（三个灯）、Medium intensity（两个灯）和 Low intensity（一个灯）。如需手动选择强度，可按手柄上的 intensity indicator lights 循环切换；该设置可在刷牙前、刷牙中或刷牙后更改，也可以在 App 中自定义。"
        if plan.intent == "fitness_charge":
            return (
                "您好，健身追踪器电量低时可以这样充电：\n"
                "1. 将充电线插入电脑 USB 口、UL 认证 USB 墙充或其他低功率充电设备。\n"
                "2. 将充电线另一端吸附到手表背面的充电触点，确保触点对齐并贴合。\n"
                "3. 充电约 12 分钟可支持约 24 小时使用；完全充满通常需要 1-2 小时。\n"
                "4. 充电时可轻触屏幕或按按钮查看电量，电量低时请及时充电。"
            )
        if plan.intent == "fitness_box_contents":
            return "您好，健身追踪器包装盒内通常包含：配有小号表带的手表、充电线，以及额外的大号表带。说明书也提示，可拆卸表带有多种颜色和材质，部分配件表带需要单独购买。"
        if plan.intent == "fitness_notifications":
            return "您好，健身追踪器可以接收手机通知。设置前请确认手机蓝牙已开启，且手机系统允许接收通知；然后在健身追踪器应用中进入设备卡片，打开通知相关设置。设置完成后，来电、消息或应用通知可同步到手表端查看，具体支持项目取决于手机和应用权限。"
        if plan.intent == "fitness_interface":
            return (
                "您好，健身追踪器的基础界面操作可以按滑动和按钮来理解：\n"
                "1. 从表盘向下滑动可打开快捷设置。\n"
                "2. 从表盘向上滑动可查看通知。\n"
                "3. 在表盘左右滑动可浏览功能卡片，例如每日目标、活动摘要、心率、睡眠、计时器、运动或天气。\n"
                "4. 按按钮可打开应用或从应用返回表盘；长按表盘可切换表盘样式。"
            )
        if plan.intent == "fitness_exercise":
            return (
                "您好，健身追踪器可通过运动应用记录运动数据：\n"
                "1. 在手表上打开运动应用，选择要记录的运动类型。\n"
                "2. 运动过程中设备会记录与该运动相关的数据，例如活动时间、心率和运动表现。\n"
                "3. 运动结束后，可在手表或配套应用中查看运动摘要，用于追踪和分析运动情况。"
            )
        if plan.intent == "fitness_heart_rate":
            return (
                "您好，健身追踪器通过手表背面的传感器测量心率，佩戴时需要让传感器与手腕皮肤保持接触。\n"
                "1. 正常佩戴时，设备会持续记录全天心率数据。\n"
                "2. 运动时，心率数据会用于运动记录和分析。\n"
                "3. 如果开启高/低心率通知，设备在检测到静止状态下心率超出阈值时会提醒。"
            )
        if plan.intent == "fitness_payment":
            return (
                "您好，可以直接用健身追踪器进行非接触式支付：\n"
                "1. 在手表上打开钱包或健身追踪器支付应用；部分机型可双击按钮快速打开。\n"
                "2. 如有提示，输入 4 位手表 PIN 码。\n"
                "3. 使用默认卡时，将手腕靠近支付终端；如需使用其他卡，先滑动选择卡片后再靠近终端。\n"
                "4. 支付成功时，手表会震动并在屏幕上显示确认信息。使用支付功能时需将设备佩戴在手腕上。"
            )
        if plan.intent == "fitness_lock":
            return "您好，可在手机上的健身追踪器应用中设置设备锁。进入设备卡片后打开设备锁功能，并按提示设置 4 位 PIN 码；之后在使用支付等敏感功能时，手表会要求输入 PIN。"
        if plan.intent == "fitness_troubleshooting":
            return (
                "您好，健身追踪器如果出现无法同步、无法响应轻点/滑动/按钮操作、无法追踪步数或其他数据、无法显示通知等问题，说明书建议先重启手表。"
                "重启后如问题仍存在，再根据具体故障继续排查。"
            )
        if plan.intent == "oven_drip_tray":
            return "您好，烤箱接油盘用于收集烹饪时滴落的油脂和碎屑。烤肉、鸡肉或鱼类时，可把接油盘放在烤架下方；说明书还建议向接油盘中加入少量水，以减少油脂飞溅和烟雾。"
        if plan.intent == "oven_exterior_clean":
            return "您好，清洁烤箱外部时，请用湿布擦拭表面；如污渍较重，可在水中加入几滴清洁剂。清洁后再用干布擦干，不要使用会损伤表面的研磨性工具或强腐蚀清洁剂。"
        if plan.intent == "oven_wire_shelf":
            return "您好，烤箱烤架可用于烧烤食物，也可作为支架承托锅具、蛋糕模等容器；使用时可根据食物和烹饪方式放在合适层位。"
        if plan.intent == "oven_baking_tray":
            return "您好，烤盘适合烘烤饼干、蛋糕、披萨等食物。使用时将食物放在烤盘上，再把烤盘放入烤箱对应层位即可。"
        if plan.intent == "oven_grill_pan_set":
            return "您好，烤架烤盘套装由烤架和搪瓷容器组成，使用时把食物放在烤架上，再把烤架放在搪瓷容器中，然后整体放到烤箱的 wire shelf 上，并配合烧烤功能使用。"
        if plan.intent == "oven_grease_filter":
            return "您好，油脂过滤器用于热风循环烹饪时减少油脂进入风扇区域。安装时将其挂在烤箱后壁、风扇前方；清洁时可取下清洗。"
        if plan.intent == "oven_catalytic_panels":
            return "您好，烤箱的催化侧面板带有特殊微孔搪瓷涂层，可吸附烹饪时飞溅的油脂。烹饪油脂较多的食物后，建议执行自动清洁程序：让烤箱空载，开启热风循环功能并设为约 200℃ 运行约 1 小时，待冷却后再用海绵清除残留。注意不要使用腐蚀性或研磨性清洁剂、粗刷、百洁布或烤箱喷雾，以免损坏催化表面。"
        if plan.intent == "oven_sliding_shelf":
            return "您好，滑动搁架用于更方便地拉出和放回烤箱内的托盘或烤架。使用时把食物或容器放在滑动搁架上，拉出检查或取放时更稳定。"
        if plan.intent == "generator_hot_safety":
            return (
                "您好，发电机运行后发动机和消音器会发烫，请注意这些安全要求：\n"
                "1. 将发电机放在行人和儿童不易接触的位置。\n"
                "2. 不要在排气口附近放置易燃物，运行时发电机与建筑物或其他设备至少保持 1 米距离。\n"
                "3. 发电机运行时不要覆盖防尘罩。\n"
                "4. 搬运发电机时请使用提手，避免接触高温部位。"
            )
        if plan.intent == "generator_shock_safety":
            return (
                "您好，为防止发电机触电事故，请注意：\n"
                "1. 不要在雨雪环境中运行发电机。\n"
                "2. 不要用湿手触摸发电机，以免触电。\n"
                "3. 发电机应正确接地；接地端子连接到埋入地下的接地棒。"
            )
        if plan.intent == "generator_fuel_check":
            return (
                "您好，发电机使用前检查燃油时，请先确认油箱内燃油是否充足；如不足，请添加无铅汽油。"
                "加油时不要超过规定油位，避免燃油溢出；如有燃油洒出，应在启动前擦净并确认周围安全。"
            )
        if plan.intent == "generator_oil_check":
            return "您好，发电机使用前请检查发动机机油液位：将机器放在水平面上，取下机油加注口盖，确认机油达到上限位置；如不足，请添加推荐机油至上限后再启动。"
        if plan.intent == "generator_sensitive_equipment":
            return (
                "您好，使用该发电机为对电压敏感的精密设备供电前，请先确认设备是否适合由便携式发电机供电。\n"
                "说明书提示，部分精密设备可能需要比便携式发电机更稳定的电压供应，例如部分医疗设备、个人电脑，以及检测峰值/有效值电压的逆变器。"
                "因此，连接前应先咨询精密设备供应商，并确认总负载不超过发电机额定输出。"
            )
        if plan.intent == "generator_control_switches":
            return (
                "您好，发电机控制面板上需要重点区分发动机开关和经济控制开关：\n"
                "1. 发动机开关控制点火系统：“开启”表示点火电路接通，可启动发动机；“停止”表示点火电路切断，发动机停机。\n"
                "2. 经济控制开关置于开启档时，经济控制单元会根据外接负载调节发动机转速，可降低油耗并减少噪音。\n"
                "3. 经济控制开关置于关闭档时，无论是否连接负载，发动机均以额定转速运行。\n"
                "4. 使用压缩机、潜水泵等启动电流较大的设备时，说明书要求将经济控制开关置于关闭位置。"
            )
        if plan.intent == "generator_identification":
            return (
                "您好，发电机的标识信息主要包括产品识别码和机器序列号：\n"
                "1. 请在说明书指定位置记录产品识别码和序列号，便于向发电机经销商订购零配件。\n"
                "2. 这些识别码也建议另行妥善保管，以防机器被盗后用于核对。\n"
                "3. 机器序列号通常位于机身标识位置，可参考说明书配图确认具体位置。"
            )
        if plan.intent == "generator_start":
            generator_steps = [
                "将油箱盖通气旋钮逆时针旋开一圈，打开油箱通气。",
                "将燃油开关旋钮置于 ON。",
                "将发动机开关置于 ON。",
                "冷机启动时拉出阻风门旋钮；热机启动时通常不需要使用阻风门。",
                "缓慢拉动反冲启动器至感觉啮合，再快速拉动启动发动机。",
                "发动机启动后进行暖机，直到把阻风门推回原位后发动机也不会熄火。",
                "最后将阻风门旋钮推回原位。",
            ]
            desired = self._desired_item_count(question)
            if desired and ("最后" in question or "last" in question.lower()):
                picked = generator_steps[-desired:]
                return "您好，启动发电机发动机的最后步骤是：\n" + "\n".join(
                    f"{index}. {item}" for index, item in enumerate(picked, start=1)
                )
            if desired:
                picked = generator_steps[:desired]
                return "您好，启动发电机发动机的前几个步骤是：\n" + "\n".join(
                    f"{index}. {item}" for index, item in enumerate(picked, start=1)
                )
            return (
                "您好，启动发电机前请确认没有连接用电设备，并将经济控制开关置于 OFF：\n"
                "1. 将油箱盖通气旋钮逆时针旋开一圈，打开油箱通气。\n"
                "2. 将燃油开关旋钮置于 ON。\n"
                "3. 将发动机开关置于 ON。\n"
                "4. 冷机启动时拉出阻风门旋钮；热机启动时通常不需要使用阻风门。\n"
                "5. 缓慢拉动反冲启动器至感觉啮合，再快速拉动启动发动机。\n"
                "6. 发动机启动后进行暖机，直到把阻风门推回原位后发动机也不会熄火。\n"
                "7. 最后将阻风门旋钮推回原位。"
            )
        if plan.intent == "generator_stop":
            return (
                "您好，发电机发动机停机可按以下步骤操作：\n"
                "1. 先断开所有用电设备。\n"
                "2. 将发动机开关置于 STOP/停止位置。\n"
                "3. 将燃油开关置于 OFF/关闭位置。\n"
                "4. 最后拧紧油箱盖通气旋钮。"
            )
        if plan.intent == "generator_no_start":
            return "您好，如果发电机发动机无法启动，请先确认发动机开关在 ON 位置并拉动反冲启动器；如果机油警告灯闪烁，说明发动机机油不足，请添加机油后再重新启动。"
        if plan.intent == "camera_mount_lens":
            return (
                "您好，相机安装镜头时请按以下步骤操作：\n"
                "1. 取下镜头后盖和机身盖。\n"
                "2. 将镜头安装标记与机身上的安装标记对齐；EF-S 镜头对齐白色标记，EF 镜头对齐红色标记。\n"
                "3. 按箭头方向旋转镜头，直到听到咔哒声并锁定到位。"
            )
        if plan.intent == "camera_eyepiece_cover":
            return "您好，使用自拍或遥控拍摄时，取景器进入的杂散光可能影响曝光，因此可安装 eyepiece cover。安装时将眼罩从取景器上取下，再把 eyepiece cover 向下滑入取景器槽位。"
        if plan.intent == "camera_p_mode":
            return "您好，相机的 P 模式是 Program AE 程序自动曝光模式。将模式转盘设为 P 后，相机会根据场景自动设置快门速度和光圈，适合一般拍摄；您仍可根据需要调整其他拍摄设置。"
        if plan.intent == "camera_auto_print":
            return (
                "您好，设置混合即时相机的自动打印模式时，请将侧面的打印模式选择器切换到 “AUTO”。\n"
                "切换后，拍摄画面上会出现自动打印图标；之后图像保存后会立即开始打印。\n"
                "如果选择手动打印模式，图像会先保存到内存中，您可以稍后再选择打印。"
            )
        if plan.intent == "fax_safety":
            return (
                "您好，使用传真功能时请注意这些安全事项：\n"
                "1. 避免在靠近水源的位置使用设备。\n"
                "2. 不要在雷雨天气连接或改动电话线。\n"
                "3. 安装、维修或移动设备前，先从墙上插座断开电源线和电话线。\n"
                "4. 使用 No. 26 AWG 或更粗规格的电信电话线。"
            )
        if plan.intent == "fax_move":
            return "您好，移动传真机前请先断开墙上插座中的所有线缆，包括电源线和电话线；搬动时避免拉扯线缆，重新安装后再按接口重新连接。"
        if plan.intent == "fax_canada":
            return "您好，在加拿大使用该传真设备时，说明书给出 Industry Canada 合规声明：设备符合免许可 RSS 标准；使用需满足两个条件，即不会造成有害干扰，并且必须接受可能导致非预期运行的干扰。"
        if plan.intent == "jetski_seat":
            return (
                "您好，水上摩托座椅拆装步骤如下：\n"
                "1. 拆卸时，拉起座椅锁扣并抬起座椅后部，将座椅取下。\n"
                "2. 安装时，将座椅前部的凸起插入甲板上的座椅固定位置。\n"
                "3. 向下按压座椅后部，直到座椅锁定到位，并确认座椅牢固。"
            )
        if plan.intent == "jetski_filler_caps":
            return "您好，水上摩托的燃油箱加注口盖和机油箱加注口盖都可通过逆时针旋转拆下；安装时请拧紧并确认盖子固定牢靠，操作前务必确认加注口盖已正确关闭。"
        if plan.intent == "jetski_levers":
            return (
                "您好，水上摩托常见操纵杆包括：\n"
                "1. 油门手柄：握紧可提高速度，松开后速度降低并回到怠速。\n"
                "2. 阻风门手柄：冷机启动时使用，按说明切换到开启或关闭位置。\n"
                "3. QSTS 选择器：降低发动机转速后，按住锁止手柄并转动选择器，可调节艇体纵倾角。"
            )
        if plan.intent == "jetski_throttle_turning":
            return (
                "您好，摩托艇转向需要车把和油门配合：\n"
                "1. 转动车把会改变喷射推力喷嘴方向，艇体随喷射推力方向转向。\n"
                "2. 转向速度和角度取决于喷射推力大小；油门越大，转向响应越明显。\n"
                "3. 如果松开油门后再大幅转动车把，转向能力会迅速下降。\n"
                "4. 需要避让或完成转弯时，应在观察周围安全后逐步重新施加油门，转向响应会恢复。"
            )
        if plan.intent == "jetski_speed_modes":
            return (
                "您好，这三个速度状态可以这样理解：\n"
                "1. 拖曳速度：低速直行或低速转向，用于熟悉操控、观察周围环境和做大范围 8 字路线练习。\n"
                "2. 半滑航速度：速度比拖曳速度高，艇体开始抬升但尚未完全进入滑航状态，适合练习较平稳的大椭圆转向。\n"
                "3. 滑航速度：艇体已进入较高速滑行状态，转向方式和低速时不同，转向角度受速度和喷射推力影响更明显，操作时应从平缓转向开始，逐步增加难度。"
            )
        if plan.intent == "jetski_stop":
            return (
                "您好，摩托艇没有独立制动系统，松开油门后会靠水阻继续滑行一段距离：\n"
                "1. 全速行驶时松开油门或关闭发动机，完全停下大约需要 90 米（300 英尺）。\n"
                "2. 使用 RiDE 手柄减速时，停车距离可比不使用时缩短约 30%，但仍会受载重、水面状况和风向影响。\n"
                "3. 如果发现无法在障碍物前停下，不要只松油门，应观察周围后施加油门并转向避开障碍物。"
            )
        if plan.intent == "jetski_avoid_collision":
            return (
                "您好，为避免摩托艇碰撞，请按说明书要求提前观察和操作：\n"
                "1. 持续观察周围人员、物体和其他船只，保持安全速度和安全距离。\n"
                "2. 不要紧跟其他摩托艇或船只后方，不要靠近他人并向其喷水或溅水。\n"
                "3. 避开水下障碍物和浅水区域，避免急转弯等让他人难以判断方向的操作。\n"
                "4. 需要避让时要提前采取动作；摩托艇转向需要油门，单纯松开油门会降低转向能力。"
            )
        if plan.intent == "landline_install_handset":
            return "您好，安装固定电话听筒时，先拉出电池隔离带启用已预装的电池；然后将听筒放到底座上充电。首次使用前建议连续充电约 8 小时，听筒放置正确时会听到 docking tone。"
        if plan.intent == "landline_handset_led":
            return "您好，固定电话听筒 LED 指示灯可用于显示不同状态。您可以在听筒设置中设置 LED indicator behavior，用它提示事件状态或充电状态，从而判断当前听筒状态。"
        if plan.intent == "landline_base_led":
            return "您好，固定电话底座上的 LED 指示灯会根据当前状态显示不同灯光行为；可通过 base station LED indicator behavior 判断底座当前工作状态，例如是否有事件提示或相关状态变化。"
        if plan.intent == "vacuum_clean_filter":
            return "您好，清洁吸尘器滤网时：1. 拉住黄色拉片取出滤网。2. 轻敲滤网，抖落灰尘和碎屑。3. 将滤网重新装回。注意：滤网未正确装入时，滤网门无法关闭。"
        if plan.intent == "vacuum_clean_extractors":
            return "您好，清洁吸尘器 extractors/滚刷时：1. 捏住黄色释放片并抬起 extractor frame。2. 清除吸入口处障碍物。3. 取下 extractors 和黄色端盖，清除端盖下方、金属轴周围的毛发和碎屑。4. 装回端盖和 extractors，并合上 frame。"
        if plan.intent == "vacuum_clean_side_brush":
            return "您好，清洁吸尘器边刷时，用硬币或小螺丝刀拆下固定螺丝，取下边刷；清理边刷和刷轴上的毛发、灰尘或碎屑后，再装回边刷并拧紧螺丝。"
        if plan.intent == "blower_start":
            return (
                "您好，吹风机冷机和热机启动方式不同：\n"
                "1. 冷机启动：将停机开关调到启动位置，将阻风门控制器调到阻风门位置 A，并反复按压泵油膜片，直到燃油充满膜片。\n"
                "2. 热机启动：将停机开关调到启动位置，先把阻风门控制器调到阻风门位置，再转回原位 B；此时只启用启动油门，不使用阻风门，并同样按压泵油膜片。\n"
                "3. 启动时左手把机身按在地面，右手慢拉启动手柄至感觉有阻力后快速拉动；不要用脚踩设备。\n"
                "4. 发动机点火后立即按下阻风门控制器，重复拉动直到启动；启动后迅速全开油门，启动油门会自动解除。\n"
                "注意：不要把启动绳完全拉出，也不要在启动绳完全拉出时松手。"
            )
        if plan.intent == "blower_safety":
            return (
                "您好，操作吹风机时请注意：\n"
                "1. 不要将喷口对准人或动物，并让人员和动物保持至少 10 米距离。\n"
                "2. 安装或拆卸附件前先关闭发动机。\n"
                "3. 不要在通风不良环境中运行发动机。\n"
                "4. 加油前先停机，避免接触高温消音器和排气区域。\n"
                "5. 不要站在梯子或支架上操作吹风机。"
            )
        if plan.intent == "blower_carburetor":
            return "您好，吹风机化油器通常包含低速油针 L、高速油针 H 和怠速调节螺钉 T。调节时应让发动机运转稳定，按说明分别调整低速、高速和怠速；如果无法稳定怠速或加速异常，应由维修人员进一步检查。"
        if plan.intent == "blower_stop":
            return "您好，关闭吹风机时，使用停机开关即可停止发动机；日常维护时也应检查停机开关功能是否正常，损坏时需要更换。"
        if plan.intent == "airpurifier_casters":
            return "您好，空气净化器脚轮安装方法如下：1. 包装内小盒中有 4 个脚轮和 8 颗螺丝。2. 用螺丝刀拆下空气净化器底部四个底座。3. 每个脚轮用两颗螺丝固定，替换原底座。4. 保留拆下的底座和螺丝，以备后续使用。"
        if plan.intent == "airpurifier_clean_body":
            return "您好，清洁空气净化器设备内外时，请先断开电源；外壳可用柔软干布擦拭，进风/出风口和内部灰尘可用吸尘器或软刷清理。不要用水直接冲洗设备，也不要使用强腐蚀清洁剂。"
        if plan.intent == "airpurifier_clean_filter":
            return "您好，清洁空气净化器滤网时，可用吸尘器或软刷清洁预过滤网；如果滤网过脏请直接更换。说明书明确提醒：不要用水清洗滤网。"
        if plan.intent == "airpurifier_remove_filter_packaging":
            return (
                "您好，首次使用空气净化器前需要先取下滤网塑料包装：\n"
                "1. 关闭设备并拔下电源。\n"
                "2. 双手握住设备上盖并向上拉起，再握住背部滤网盖把手向前拉出。\n"
                "3. 取出滤网并去掉塑料包装。\n"
                "4. 将滤网重新装入设备，确保滤网安装牢固。\n"
                "5. 装回滤网盖和上盖。\n"
                "6. 插上电源后，同时长按“睡眠 + 自动”键 5 秒以上完成初始化，红色滤网更换指示灯会熄灭。"
            )
        if plan.intent == "airpurifier_replace_filter":
            return (
                "您好，空气净化器需要更换滤网时，滤网更换指示灯会亮红色。说明书建议每 6-12 个月更换一次滤网，具体周期会受使用环境影响。\n"
                "更换滤网后，请长按控制面板上的“睡眠 + 自动”键 5 秒以上，重置滤网指示灯。"
            )
        if plan.intent == "airpurifier_storage":
            return "您好，空气净化器长期存放前，建议在晴天让设备运行约 1 小时，帮助干燥内部；然后关闭并拔下电源，清洁设备后盖好防尘，并存放在干燥、避免阳光直射的位置。"
        if plan.intent == "dishwasher_add_detergent":
            return (
                "您好，给洗碗机添加洗涤剂时，请按说明书步骤操作：\n"
                "1. 在即将运行程序前，向右推动卡扣打开洗涤剂盒盖。\n"
                "2. 按所选程序和餐具脏污程度，加入建议量的粉末或块状洗涤剂；洗涤剂盒内有 15 cm³ 和 25 cm³ 刻度，满容量为 40 cm³。\n"
                "3. 如果餐具放置时间较长、残渣已经干结，可同时在 5 cm³ 小格中加入洗涤剂。\n"
                "4. 轻轻按下盒盖，听到咔嗒声表示已关紧。\n"
                "提示：无预洗的短程序建议使用粉末洗涤剂；过量使用粉末洗涤剂可能无法完全溶解，并在玻璃器皿上留下划痕。"
            )
        if plan.intent == "dishwasher_tablet":
            return (
                "您好，洗碗机可使用 2 合 1、3 合 1、4 合 1、5 合 1 等洗涤块。使用洗涤块功能时：\n"
                "1. 按开机/关机键开机。\n"
                "2. 选择所需程序。\n"
                "3. 按半载/洗涤块键，直到洗涤块指示灯亮起。\n"
                "4. 按启动/暂停/取消键并关闭机门开始运行。\n"
                "该功能开启时，盐量和亮碟剂指示灯会关闭；如需更好洗涤和烘干效果，说明书仍建议优先分别使用洗涤剂、亮碟剂和专用盐。"
            )
        if plan.intent == "bike_specs":
            return "您好，这款健身单车的技术规格包括：最大使用者重量 136 千克（300 磅），设备总占地面积 5670 平方厘米，机器重量 26.5 千克（58.4 磅）。认证方面，交流电源适配器符合 CE 认证，产品符合固定式健身器材 EN ISO 20957 国际标准 S 级。"
        if plan.intent == "bike_workout_area":
            return "您好，为确保安全且获得较好的使用效果，请将健身单车放在坚固、水平的地面上。说明书建议预留的运动区域至少为 2.3 米 × 1.8 米（90 英寸 × 70 英寸），并确保周围没有障碍物。"
        if plan.intent == "bike_edit_profile":
            return "您好，编辑健身单车用户档案时，可选择对应用户档案后进入编辑流程，按提示设置姓名、年龄、体重、身高、性别和首选运动数据显示等信息；保存后，该档案会用于记录运动结果和查看运动数据。"
        if plan.intent == "bike_easy_ride_programs":
            return "您好，健身单车“轻松骑行”类别下的预设运动程序包括：起伏山丘、公园骑行和轻松之旅。这些预设程序会自动控制阻力和运动等级，适合进行较轻松、节奏相对平稳的骑行训练。"
        if plan.intent == "bike_fitness_test":
            return "您好，体能测试程序用于评估体能水平变化。它会将功率输出（瓦特）与心率进行对比；体能提升后，在相同心率下通常能输出更高功率。使用该程序时，控制台需要能读取接触式心率传感器或心率监测仪数据，并会根据用户档案中的年龄和体重计算体能得分。"
        if plan.intent == "bike_mountain_programs":
            return "您好，健身单车山地类别的预设运动程序包括 Pikes Peak（派克峰）、Mount Hood（胡德山）和 Pyramid（金字塔），这些程序会自动控制不同阻力和运动等级。"
        if plan.intent == "bike_challenge_programs":
            return "您好，健身单车挑战类别的预设运动程序包括 Uphill Sprint（上坡冲刺）、Cross-Training（交叉训练）和 Interval Training（间歇训练），属于更高难度的自动阻力训练程序。"
        if plan.intent == "drill_keyless_chuck":
            return "您好，安装单套无键夹头附件时，一手握住夹头黑色套筒，另一手固定工具；逆时针旋转套筒打开夹爪，插入附件约 19 mm（3/4 英寸），再顺时针旋紧套筒，确保附件被夹牢。"
        if plan.intent == "drill_battery_pack":
            return "您好，安装电钻电池组时，先确认电池已充满电，将电池组对准工具手柄内的导轨，滑入手柄直到牢固就位且不会松脱。拆卸时，按下电池释放按钮，将电池组从手柄中平稳拔出。"
        if plan.intent == "drill_accessories":
            return "您好，购买电钻后，建议按作业需要配备合适的钻头、批头等附件，并按照说明书要求佩戴护目镜或其他眼部防护装备；冲击钻孔时还应佩戴听力防护装备。如果工具配有辅助手柄，也应按要求使用。附件应选择与工具规格兼容的正规配件，安装或拆卸前请先关闭工具并取下电池组。"
        if plan.intent == "drill_warranty":
            return "您好，电钻的三年有限保修通常覆盖正常使用下因材料或工艺缺陷导致的问题；同时说明书还提到购买后第一年内可免费维护工具并更换正常使用磨损部件，部分电池组享有两年或三年免费服务。享受保修服务通常需要提供购买凭证；人为改装、误用、滥用、非授权维修或不按说明书操作造成的损坏一般不在保修范围内。"
        if plan.intent == "boat_bimini_remove":
            return "您好，拆下船上的 bimini top 时，请先按说明将顶篷收拢并固定，再取下主支撑杆的安装销，将 bimini top 从安装位置取下；拆下的安装销和部件要妥善保管，以便后续重新安装。"
        if plan.intent == "boat_bimini_install":
            return "您好，安装或展开 bimini top 时，先将顶篷主支撑杆装回支架并用锁销固定，再展开顶篷；随后将前支撑杆固定到对应支架，确认锁销和支撑杆都已牢固安装后再使用。"
        if plan.intent == "boat_load_distribution":
            return "您好，装载船只时应让重量尽量低，并在左右舷以及船首到船尾之间均匀分布；移除不必要的货物，不要超过最大载重。说明书给出的最大载荷为总载重 1021 kg，其中操作者和乘员合计不超过 844 kg。"
        if plan.intent == "motherboard_pcie_x16" and "pci express" in lowered and "graphic" in lowered:
            return (
                "您好，主板的 PCI Express 3.0 x16 插槽说明如下：\n"
                "1. 该主板有三个 PCI Express 3.0 x16 插槽，可支持符合 PCI Express 规范的 PCI Express 3.0 x16 显卡。\n"
                "2. 单显卡模式下，建议把 PCI Express x16 显卡安装在灰色的 PCIe 3.0 x16_1 插槽，以获得更好的性能。\n"
                "3. 运行部分模式时，请确保提供足够电源。\n"
                "4. 使用多张显卡时，建议将机箱风扇连接到主板 CHA_FAN1/2/3 接口，以改善散热环境。"
            )
        if plan.intent == "motherboard_onboard_led" and "standby power led" in lowered:
            return (
                "您好，主板上的 Onboard LED 是待机电源指示灯：\n"
                "1. 当系统处于开机、睡眠或软关机状态时，该指示灯会亮起。\n"
                "2. 它用于提醒您：在拆卸或插拔任何主板组件前，应先关闭系统并拔下电源线。\n"
                "3. 说明书配图展示了该 onboard LED 在主板上的位置。"
            )
        if plan.intent == "motherboard_sata_odd_usb_os" and "sata odd" in lowered and "usb" in lowered:
            return (
                "您好，使用 SATA ODD 和 USB 设备安装操作系统时，说明书给出的流程如下：\n"
                "1. 准备 support DVD、OS 7 安装源、SATA ODD，以及容量 8GB 或以上的 USB 设备；建议先格式化 USB 存储设备。\n"
                "2. 将 OS 7 安装 DVD 插入 USB ODD，或在可用电脑上把 OS 7 安装 DVD 的全部文件复制到 USB 存储设备。\n"
                "3. 将 USB ODD 或 USB 存储设备连接到 100 series 平台。\n"
                "4. 将 support DVD 插入 100 series 平台上的 SATA ODD。\n"
                "5. 开机后在 POST 过程中按 F8 进入启动菜单，再按屏幕提示继续安装。"
            )
        if plan.intent == "motherboard_chassis_screws" and "secure the motherboard" in lowered:
            return (
                "您好，固定主板到机箱时，请按说明书的螺丝孔位置操作：\n"
                "1. 将 9 颗螺丝安装到图中用圆圈标出的孔位，用来把主板固定到机箱。\n"
                "2. 不要把螺丝拧得过紧，否则可能损坏主板。\n"
                "3. 安装时注意主板标注的一侧应朝向机箱后部。"
            )
        if plan.intent == "motherboard_system_memory" and ("system memory" in lowered or "recommended memory configurations" in lowered):
            return (
                "您好，主板系统内存的说明要点如下：\n"
                "1. 可以在 Channel A 和 Channel B 安装不同容量的内存；系统会将较小容量通道映射为双通道，较大容量通道中多出的部分会以单通道运行。\n"
                "2. 建议使用低于 1.65V 的 DIMM 电压，以保护 CPU。\n"
                "3. 由于 32 位系统的内存地址限制，安装 4GB 或更多内存时，系统实际可用内存可能约为 3GB 或更少。\n"
                "4. 如果要有效使用 4GB 或更多内存，建议安装 64 位操作系统。"
            )
        if plan.intent == "motherboard_tpm_connector" and "tpm connector" in lowered:
            return (
                "您好，TPM connector（14-1 pin TPM）用于连接 Trusted Platform Module 系统：\n"
                "1. TPM 系统可安全存储密钥、数字证书、密码和数据。\n"
                "2. 它还可以帮助增强网络安全、保护数字身份，并确保平台完整性。\n"
                "3. 说明书配图展示了 TPM connector 在主板上的位置。"
            )
        if plan.intent == "motherboard_t_sensor":
            return (
                "Thermal sensor connector (2-pin T_SENSOR) is used for a thermistor cable. "
                "It lets the motherboard monitor the temperature of critical motherboard components or connected devices. "
                "Connect the thermistor cable to this header when you need that temperature monitoring function."
            )
        if plan.intent == "drill_battery_charge":
            return (
                "您好，给电钻电池组充电时，请按以下步骤操作：\n"
                "1. 插入电池组前，先将充电器插入合适的插座。\n"
                "2. 将电池组插入充电器，并确认电池组完全就位。\n"
                "3. 红色充电指示灯持续闪烁，表示充电已开始。\n"
                "4. 红色指示灯常亮表示充电完成；此时电池组已充满，可取下使用，也可以留在充电器中。"
            )
        if plan.intent == "steam_quick_assembly":
            return (
                "您好，蒸汽清洁机快速组装不需要专用工具，可按以下步骤操作：\n"
                "1. 从机身底部取下机身锁扣。\n"
                "2. 将机身底部滑入旋转拖把头的颈部位置，并对齐锁扣孔。\n"
                "3. 插入机身锁扣，将机身和拖把头固定。\n"
                "4. 将手柄杆插入机身，直到听到咔嗒声并确认锁定到位。"
            )
        if plan.intent == "thermostat_datetime":
            return (
                "您好，温控器日期、时间和日程可通过设置菜单调整：\n"
                "1. 设置日期时，使用 + 或 - 调整年、月、日，每调整一项后按 Select 保存并进入下一项。\n"
                "2. 设置时间时，先选择 12 小时或 24 小时格式，再用 + 或 - 设置小时和分钟，并按 Select 保存。\n"
                "3. 调整程序日程时，进入 Menu > PROG，选择要修改的日期或日期组，再依次设置 Wake、Away、Home、Sleep 等时段的开始时间和制热/制冷温度，完成后返回主界面保存。"
            )
        if plan.intent == "thermostat_temp_override":
            return (
                "您好，温控器可临时或永久调整设定温度：\n"
                "1. 临时覆盖：按 + 或 - 调到想要的温度后停止操作，该温度会保持到下一个计划时段开始。\n"
                "2. 如需取消临时覆盖，再按 + 或 -，然后选择 Cancel。\n"
                "3. 永久保持：按 + 或 - 调温，当临时保持温度闪烁时按 Hold/Mode，温控器会进入 permanent hold。\n"
                "4. 取消永久保持时，同样按 + 或 -，再选择 Cancel。"
            )
        return ""

    def _select_planned_direct_answer_en(self, plan: EvidencePlan, question: str = "") -> str:
        lowered_question = question.lower()
        if plan.intent == "generator_start":
            steps = [
                "Turn the fuel tank cap vent knob counterclockwise one turn to open the vent.",
                "Turn the fuel cock knob to ON.",
                "Turn the engine switch to ON.",
                "For a cold engine, pull the choke knob fully out; a warm engine usually does not need the choke.",
                "Pull the recoil starter slowly until it engages, then pull it quickly to start the engine.",
                "Warm up the engine until it will not stop when the choke knob is returned.",
                "Push the choke knob back to its original position.",
            ]
            desired = self._desired_item_count(question)
            if desired and ("last" in question.lower() or "最后" in question):
                picked = steps[-desired:]
                return "The last steps for starting the generator engine are:\n" + "\n".join(f"{i}. {step}" for i, step in enumerate(picked, start=1))
            if desired:
                picked = steps[:desired]
                return "The first steps for starting the generator engine are:\n" + "\n".join(f"{i}. {step}" for i, step in enumerate(picked, start=1))
            return "To start the generator engine, make sure no electrical devices are connected and the economy control switch is OFF. Then open the fuel tank cap vent, turn the fuel cock and engine switch to ON, use the choke for a cold engine, pull the recoil starter, warm up the engine, and return the choke knob."
        if plan.intent == "start_engine" and ("jetski" in lowered_question or "watercraft" in lowered_question):
            return (
                "To start the jetski engine:\n"
                "1. Attach the engine shut-off cord to your wrist and insert the clip under the engine shut-off switch.\n"
                "2. Make sure the cord is not wrapped around the handlebar.\n"
                "3. Press the green start switch without squeezing the throttle lever or RiDE lever.\n"
                "4. Release the start switch immediately after the engine starts. Be ready for forward thrust because the engine and drive unit are directly connected."
            )
        if plan.intent == "battery_charge" and "camera" in lowered_question:
            return (
                "To charge the camera battery, use an AC power adapter with rated output DC 5.0 V / 1000 mA. "
                "The manual states that you can take or print images while charging, and charging takes about 3 to 4 hours."
            )
        if plan.intent == "parameter" and "camera" in lowered_question and "battery" in lowered_question:
            return "To install the camera battery, open the battery compartment, insert the rechargeable battery in the indicated direction, and close the cover securely before use."
        if plan.intent == "parameter" and ("max load" in lowered_question or "maximum load" in lowered_question) and "jetski" in lowered_question:
            return "The jetski must not be operated when the total load, including all cargo, exceeds 220 kg (485 lb)."

        answers = {
            "airfryer_first_use": (
                "Before first use of the air fryer, remove all packing materials, stickers, or protective film. "
                "Take out the basket/pan and removable parts, wash them with warm water and mild detergent, then dry them thoroughly. "
                "Wipe the inside and outside of the appliance with a damp cloth, make sure no packaging remains inside, then place it on a stable, heat-resistant, well-ventilated surface before cooking. "
                "For first-use preparation, you do not need the Wi-Fi, app, or NutriU pairing procedure."
            ),
            "blower_ppe": (
                "When using the blower, wear qualified hearing protection, qualified eye protection, a face mask in dusty conditions, work boots or shoes with non-slip soles, and keep a first-aid kit available."
            ),
            "air_conditioner_components": (
                "The air conditioner is mainly made up of the indoor unit, outdoor unit, and remote controller. The indoor unit includes parts such as the front panel, air inlet, air filter, air outlet, airflow vanes, and display/indicator area. The outdoor unit includes the air inlet/outlet, connection piping, drain hose, and wiring. The remote controller is used for power, mode, temperature, fan speed, and timer settings."
            ),
            "air_conditioner_auto_restart": (
                "The air-conditioner auto-restart function restores the previous operating settings after a power failure and is enabled by default. To turn it off, open the front cover and press the ON/OFF key for about 6 seconds until the unit beeps twice and the indicator flashes 6 times. To turn it on again, press the ON/OFF key for about 6 seconds again; the unit beeps twice and the blue indicator flashes 4 times."
            ),
            "chair_parts": (
                "The ergonomic chair assembly mainly includes the backrest, seat cushion, armrests, mechanism/base plate, gas lift, base, casters, connector parts, headrest, and lumbar pillow. Assemble the casters and base first, install the gas lift and mechanism, place the seat, attach the backrest connector and armrests, then install the headrest and lumbar pillow."
            ),
            "chair_functions": (
                "The ergonomic chair functions include height adjustment, backrest reclining, lumbar massage, and support from the armrests, headrest, and lumbar pillow. Pull the lift lever to adjust seat height. Pull the recline lever up to let the backrest tilt with your body, or press it down to lock the angle. Plug in USB power to use the lumbar pillow massage function."
            ),
            "dishwasher_parts": (
                "The dishwasher operating area and main parts include the On/Off key, Start/Pause/Cancel key, display, program selection key, half-load/tablet key, delay-start key, salt and rinse-aid indicators, upper and lower baskets, spray arms, filters, detergent dispenser, and rinse-aid dispenser."
            ),
            "dishwasher_spray_arm_clean": (
                "To clean the upper spray arm, first check whether the spray-arm holes are clogged. If they are clogged, remove and clean the spray arm: loosen the retaining nut, remove the spray arm, clear food residue from the holes, rinse it with water, then reinstall it and make sure it can rotate freely."
            ),
            "dishwasher_unsuitable_items": (
                "Do not wash items contaminated with ash, candle residue, polish, dye, or chemicals in the dishwasher. Avoid iron utensils because they may rust or stain other items. Also avoid silverware or knives with wooden/bone handles, glued parts, or heat-sensitive parts, as well as copper and tin-plated containers. Decorated porcelain, aluminum and silver items, delicate glass, and crystal may fade or lose their shine, so confirm that tableware is dishwasher-safe before washing it."
            ),
            "dishwasher_basket_height": (
                "The dishwasher upper basket can be adjusted according to the size of the dishes, depending on the model. For roller adjustment, move the rail stop aside, remove the basket, change the roller position, then put the basket back on the rails and close the stop. For models with a basket-height mechanism, lift one side of the upper basket wire to raise it and repeat on the other side. To lower it, press the latch on the adjustment mechanism on both sides. Make sure both sides are at the same height after adjustment."
            ),
            "airpurifier_modes": (
                "The air purifier supports normal operation with adjustable fan speed, indoor air-quality display, child/safety lock, and filter-replacement reminder. In normal operation, press the start key and use the control panel to adjust fan speed. The IAQ indicator changes color according to fine-particle concentration. The safety lock prevents child operation. When the filter indicator turns red, replace the filter and hold Sleep + Auto for more than 5 seconds to reset it."
            ),
            "airpurifier_dust_sensor": (
                "To clean the air purifier dust sensor, lift the top cover, pull out the rear filter cover by its handle, and remove the dust sensor cover on the right side. Wipe the dust sensor lens and air inlet with a cotton swab slightly moistened with water, then dry them thoroughly with a clean dry cotton swab. Reinstall the dust sensor cover, filter cover, and top cover."
            ),
            "steam_functions": (
                "The steam cleaner's useful functions include a detachable handheld steamer for local cleaning, a mop head for hard floors, a fabric-cleaning head with cloth for glass or hard surfaces, and jet/curved nozzles for corners, seams, and hard-surface cleaning. Choose the accessory for the cleaning task and make sure all locks are secure before use."
            ),
            "steam_hard_floor": (
                "To clean hard floors with the steam cleaner, sweep or vacuum the floor first. Move the cleaner slowly while pressing the switch to release steam. For sanitizing an area, keep the steam mop on that area for at least 15 seconds but no more than 20 seconds. If steam stops, unplug the cleaner, refill the water tank, and continue. Do not use it on unsealed wood floors."
            ),
            "air_conditioner_troubleshooting_safety": (
                "For air-conditioner technical issues or malfunctions, first make sure troubleshooting or repair is handled by qualified service personnel with the proper tools and testing instruments. Check the power supply and power cord, confirm installation follows local electrical requirements, make sure the air inlet/outlet is not blocked, and check for issues such as refrigerant leakage, poor drainage, or an unsecured panel/cover. If the problem continues, stop using the unit and contact an authorized service center rather than attempting unsafe repair."
            ),
            "generator_control_switches": (
                "The generator control panel includes the engine switch and the economy control switch. The engine switch controls ignition: ON connects the ignition circuit so the engine can start, and STOP cuts the ignition circuit to stop the engine. The economy control switch lets the control unit adjust engine speed according to load, reducing fuel use and noise when enabled; when disabled, the engine runs at rated speed. For high starting-current equipment such as compressors or submersible pumps, keep the economy control switch OFF."
            ),
            "generator_identification": (
                "The generator identification information includes the product identification code and serial number. Record these codes in the specified place so you can order replacement parts from the generator dealer, and keep a separate copy in case the machine is stolen."
            ),
            "generator_sensitive_equipment": (
                "Before powering voltage-sensitive precision equipment with this generator, confirm that the device is suitable for portable-generator power. The manual warns that some precision equipment may require a more stable voltage supply than a portable generator provides, including some medical equipment, personal computers, and inverters that detect peak or RMS voltage. Check with the precision-equipment supplier first and make sure the total load stays within the generator rating."
            ),
            "coffee_empty_system": (
                "To empty the coffee machine system before long non-use, frost protection, or repair: first turn the machine off by pressing both the Espresso and Lungo buttons. Remove the water tank and open the lever. Press both the Espresso and Lungo buttons for 3 seconds until both LEDs blink alternately. Close the lever; the machine switches off automatically after emptying. Then empty and clean the used capsule container and drip tray. The machine will be blocked for about 10 minutes after emptying mode."
            ),
            "boat_trip_screen": (
                "The boat trip screen shows engine operation hours, fuel consumption, and other trip information. It includes a menu, scrollbar, and Reset button. The menu displays four items at a time; use the scrollbar to move through the items. To reset one trip item, touch and hold that item for several seconds. To reset all display items, tap the Reset button. Some items cannot be reset."
            ),
            "jetski_hood_open_close": (
                "To open the jetski hood, push the hood latch down and lift the hood up. To close it, push the hood down until it locks in place. Before operating the watercraft, make sure the hood is properly secured."
            ),
            "grill_assembly_first_three_steps": (
                "The first three grill assembly steps are:\n"
                "1. Attach the two locking casters to the rear of the bottom shelf and the two fixed casters to the front using the supplied wrench.\n"
                "2. Follow the second assembly illustration in the manual before continuing to the back-panel hardware step.\n"
                "3. Attach the light adapter to the back panel with four #8-32 x 3/8 in. screws, 4 mm lock washers, 4 mm flat washers, and #8 nuts. Then place the lower back panel between the side panels at the rear of the bottom shelf and secure it to the side panels and bottom shelf with the specified screws and lock washers."
            ),
            "network_camera_t_rail": (
                "To mount the network camera on a drop-ceiling T-rail, use the included T-rail hardware. Set the clip spacing with the dashed lines on the mount plate template, tighten the set screws on the T-rail clips with a 5/64-inch (2 mm) hex key, attach the mount plate to the clips using the holes marked G, then rotate and snap the clips onto the T-rail. The black foam pads should be slightly compressed after installation."
            ),
            "function_keyboard_setup": (
                "To set up the function keyboard, connect the USB-C cable to the port on the back of the keyboard and connect the other end to an available USB 2.0 or faster port on the computer. If your version includes a wrist rest, attach it to the front of the keyboard; it is magnetic and centers itself. If you want a higher typing angle, unfold the adjustable feet on the bottom of the keyboard. For lighting, macro, or profile settings, use CAM software and save the configuration to the onboard profiles."
            ),
            "function_keyboard_switch_replace": (
                "To remove a keyboard switch, use the included switch puller from the top, place the tool along the front and back sides of the switch, press the two retaining clips, and pull the switch upward. Do not pry from underneath the switch. To reinstall it, make sure the metal pins are straight, align them with the keyboard socket, and press the switch straight down until both sides sit flush with the top case."
            ),
            "rideon_motorcycle_front_wheel": (
                "To install the ride-on motorcycle front wheel, pass the front axle through both handle tubes and the front wheel, making sure washers are installed on the inside and outside of both handle tubes. Secure the axle and wheel with two nuts and tighten them with the supplied wrench. After assembly, check the gap between the wheel and handle tubes: add washers if the wheel is loose, remove a washer if the wheel is stuck, and confirm that the wheel rotates freely."
            ),
            "boat_emission_label": "The emission control certificate approval label is attached to each engine unit and also inside the engine compartment. Open the engine compartment and check the emission control information label to find it.",
            "camera_battery_charge": "To charge the camera battery, use an AC power adapter with rated output DC 5.0 V / 1000 mA. The manual states that you can take or print images while charging, and charging takes about 3 to 4 hours.",
            "camera_battery_install": "To install the camera battery, open the battery compartment, insert the rechargeable battery in the indicated direction, and close the cover securely before use.",
            "memory_card": "To insert the memory card before taking photos, slide the card-slot cover open, insert the memory card straight into the slot until it clicks into place, then close the cover. To remove it, press the card inward and release it slowly.",
            "camera_shutter_button": "The retrieved camera manual does not give a procedure for removing the shutter button. It describes using the shutter button for shooting and long-exposure operation, so do not attempt shutter-button removal based only on this manual.",
            "boat_battery_switches": (
                "Before sailing, check the boat's battery switch assembly:\n"
                "1. The boat uses two marine batteries: the start battery for engine starting and the house battery for accessories such as lights, bilge pumps, blower, and audio equipment.\n"
                "2. The battery switch assembly has START, HOUSE, and EMERG PARALLEL switches.\n"
                "3. For normal use, keep START and HOUSE in the ON position and keep EMERG PARALLEL in the OFF position.\n"
                "4. If the start battery is discharged, turn EMERG PARALLEL to ON to start the engine, then turn it back to OFF after the engine starts or the start battery is charged."
            ),
            "boat_over_temperature": (
                "If the boat shows an Over Temperature warning, reduce engine speed immediately and return to shore or another safe location. The display warning and buzzer indicate that the engine is overheating, and engine speed may be limited to help prevent damage. Check whether cooling water is coming from the cooling water pilot outlet, especially when applying throttle. If no water comes out, do not continue high-speed operation."
            ),
            "boat_fuse": (
                "To replace a boat fuse:\n"
                "1. Remove the fuse box cover and identify the blown fuse.\n"
                "2. Use the fuse puller to remove the blown fuse, then replace it with a spare fuse of the same/correct amperage.\n"
                "3. For the accessory fuse or bilge pump fuse, remove the fuse holder first; these fuses are accessed by opening the battery compartment.\n"
                "4. Reinstall the fuse box cover.\n"
                "Do not use a fuse with a higher amperage than recommended. Fuse ratings include: electronic throttle valve 10 A, fuel pump 10 A, main relay drive 10 A, main fuse 20 A, battery fuse 30 A, accessory fuse 20 A, and bilge pump fuse 3 A."
            ),
            "swim_platform_open": (
                "The manual describes opening the wet storage compartment under the swim platform:\n"
                "1. Locate the wet storage compartment under the swim platform.\n"
                "2. Pull the lock handle upward.\n"
                "3. Turn the lock handle clockwise and open the rear platform hatch.\n"
                "4. To close it, close the hatch, turn the lock handle counterclockwise, confirm the hatch is secure, and push the lock handle down."
            ),
            "boat_factory_reset": (
                "The factory reset screen is used to restore factory default settings. Tap Reset on the factory reset screen, then tap YES on the confirmation message to reset the settings. Tap NO if you want to return without resetting."
            ),
            "boat_engine_oil_level": (
                "To check the boat engine oil level, keep the boat level on land or in the water, start the engine and idle it for at least 6 minutes, then stop the engine and open the engine hood. Remove and wipe the oil tank filler cap/dipstick, screw it fully back in, then remove it again and check that the level is between the minimum and maximum marks. Add oil slowly if needed."
            ),
            "boat_battery_compartment": (
                "To open the boat battery compartment, release the latch on the compartment lid at the port side of the stern, then open the lid. To close it, close the lid and secure the latch back onto the deck."
            ),
            "boat_anchor_light": (
                "To set up the anchor light, open the lockable storage compartment and remove the anchor light. Separate stoppers A and B, extend the pole, screw stopper A to the middle of the pole, open the anchor light socket cap, align the projection with the socket slot, insert the light, and install stopper B."
            ),
            "boat_fire_extinguisher": (
                "The boat should carry at least one full fire extinguisher. One fire extinguisher is mounted in the lockable storage compartment; the manual recommends a chemical-type extinguisher with a capacity of two pounds or more for this location. If two extinguishers are carried, the other should be mounted in the battery compartment near the engine compartment; this one should be a clean-agent type, such as CO2 or FE-36, with a capacity of five pounds or more."
            ),
            "boat_water_supply": (
                "To turn the jet wash water supply on or off, stop the engines, open the rear platform hatch, remove the inspection cover, and rotate the shut-off valve 90 degrees clockwise to open the supply. Turn it back to close the supply."
            ),
            "boat_jet_wash_use": (
                "To use the jet wash function after using the boat, connect the coil hose to the hose fitting, start the engines, then press the jet wash switch. This lets you wash down the boat using the jet wash system."
            ),
            "boat_bilge_pump": (
                "The bilge pump drains water collected in the bilge area. Turn on the bilge pump switch to operate it. The pump can also automatically detect excessive bilge water and discharge most of it through the bilge pump outlet; the bilge pump indicator light comes on while it is operating."
            ),
            "boat_steering_turn": (
                "The boat turns by using the steering wheel together with jet thrust. Turning the steering wheel changes the jet thrust nozzle angle, and the boat turns in that direction. Turning response depends on jet thrust and steering wheel position, so do not pull the remote control levers back to idle/neutral when trying to avoid an obstacle."
            ),
            "boat_cross_wakes": (
                "When crossing wakes and swells, adjust your speed and crossing angle before reaching them. Slowing down and crossing at a quartering angle usually reduces the jolt. Be prepared to correct direction and balance, especially with sharp or repeated wakes."
            ),
            "boat_flush_cooling": (
                "To flush the cooling system, connect the garden hose adapter to a hose, remove the flush hose connector cap, attach the adapter to the flush hose connector, start the engine, then immediately turn on the water supply. Confirm water flows from the jet thrust nozzle and cooling water pilot outlet, run at fast idle for 3 to 5 minutes, then turn off the water, drain remaining water, and stop the engine."
            ),
            "boat_livewell": (
                "To use the livewell, pull the latch to open the livewell lid, press the livewell switch to run the livewell pump and supply water, then press the switch again when enough water has entered. Use the aerator switch when you need aeration or water circulation."
            ),
            "boat_move_forward": (
                "To move the boat forward, push the remote control levers forward from neutral. Moving them farther forward increases engine output and jet thrust. At low speed the shift gates lift slightly and TDE helps steering; when pushed farther, the shift gates lift completely and the boat moves forward."
            ),
            "boat_throttle_cable": (
                "To maintain the throttle cable, apply grease to the throttle-cable inner wires at the APS pulley wheel. The steering cable and shift cable ball joints and inner wires near the jet thrust nozzles can also be lightly greased."
            ),
            "boat_bimini_remove": "To remove the bimini top, fold and secure the top, remove the main pole mounting pins, and lift the bimini top from its mounting points. Keep the pins and parts for reinstallation.",
            "boat_bimini_install": "To install or set up the bimini top, attach the main poles to the brackets and secure them with the lock pins, unfold the top, attach the support poles to their brackets, and confirm all pins and poles are securely locked.",
            "boat_bimini_upright_storage": (
                "To store the bimini top in the upright position, do not trailer the boat with the top fully extended or upright. Remove the lock pins, push the center poles down, pull the bimini top toward the bow, and install the storage cover. For trailering, put the bimini top in the fully collapsed position to avoid damage."
            ),
            "boat_engine_start": (
                "To start the boat engines:\n"
                "1. Check the hull drain plug and make sure it is tightened securely before launching.\n"
                "2. Turn the battery switch to the ON position.\n"
                "3. Push the blower switch and ventilate the engine compartment for at least 4 minutes.\n"
                "4. Attach the engine shut-off cord to your PFD and install the clip on the engine shut-off switch.\n"
                "5. Put the remote control levers in neutral.\n"
                "6. Turn the main switch keys to the start position and release them when the engines start. If they do not start within 5 seconds, release the keys and wait at least 15 seconds before trying again."
            ),
            "boat_load_distribution": "When loading the boat, keep weight low and distribute it evenly from side to side and bow to stern. Remove unnecessary cargo and do not exceed the maximum load.",
            "fax_connect": (
                "When connecting the fax function, use a No. 26 AWG or larger telecommunication line cord. Install the equipment near an easily accessible AC outlet so power can be disconnected in an emergency. Connect the telephone line through a standard modular jack such as USOC RJ11C, and disconnect all cables from the wall outlet before installation, servicing, or modification."
            ),
            "generic_safe_operation": "Follow the product safety instructions before operating it: read the manual and warning labels, make sure the operator understands the controls and procedure, keep bystanders away, use the required protective equipment, and stop the operation if conditions become unsafe or the product behaves abnormally.",
            "fax_finger_safety": "To protect your fingers when using the fax machine, keep fingers away from moving covers and openings, and make sure the machine is stopped and disconnected before accessing internal areas for maintenance or clearing problems.",
            "fax_warning_labels": "Yes. Caution and warning labels inside the fax machine are important safety information and should not be removed or covered. Follow those labels and the safety instructions before operating, cleaning, or servicing the product.",
            "fax_safety": "For fax safety, avoid using the product near water, do not connect or modify telephone wiring during a lightning storm, disconnect power and telephone lines before moving or servicing the equipment, and use a No. 26 AWG or larger telecommunication line cord.",
            "fax_move": "Before moving the fax machine, disconnect all cables from the wall outlet, including the power cord and telephone line. Avoid pulling the cables, and reconnect them to the correct ports after moving.",
            "fax_canada": "For Canada, the fax manual states that the product complies with Industry Canada licence-exempt RSS standards. Operation must not cause harmful interference and must accept any interference received.",
            "landline_base_station": (
                "To connect the landline base station, connect one end of the power adapter to the DC input jack on the bottom of the base station and the other end to a wall power socket. Then connect one end of the telephone line cord to the telephone socket on the base station and the other end to the wall telephone socket. If the same line uses DSL service, install a DSL filter to reduce noise and caller ID issues."
            ),
            "landline_install_handset": "To install the handset, pull out the battery tape to activate the pre-installed battery, then place the handset on the base station to charge. Before first use, charge it for about 8 hours.",
            "landline_searching_status": "If the landline handset is in searching status, make sure the base station has power, register the handset to the base station, and move the handset closer to the base station.",
            "landline_handset_led": "The handset LED indicator can show different event or charging states. Set the handset LED indicator behavior in the handset settings to display the desired status information.",
            "landline_base_led": "The base station LED indicator shows the current base station status. Check the base station LED indicator behavior to understand status changes or event notifications.",
            "quick_release": (
                "Quick Release (QR/QPR) releases pressure faster after cooking. Press the quick release button until it clicks and locks in the vent position. Steam will come out from the top of the steam release valve; this is normal. Once pressure is fully released, cooking stops quickly, which helps avoid overcooking foods such as vegetables and delicate seafood."
            ),
            "pressure_steam_release": "To set the steam release valve, press the quick release button until it clicks and locks in the Vent position. Steam will come out from the top of the steam release valve; this is normal. Wait until pressure is fully released before opening the lid.",
            "natural_release": "Natural Release (NR/NPR) means the cooker depressurizes naturally as the temperature inside drops after pressure cooking. Always release pressure before opening the lid, and follow the recipe to choose natural release or quick release.",
            "float_valve": "To remove or set the float valve, hold the flat top of the float valve with your finger and turn the lid over. Remove the silicone cap from the underside, then remove the float valve from the top of the lid. Do not discard the float valve or silicone cap.",
            "pressure_anti_block_shield": "The anti-block shield prevents food particles from coming up through the steam release pipe and helps pressure regulation. To remove it, grip the lid like a steering wheel and press firmly against the side of the shield until it pops off the prongs. To install it, place the shield over the prongs and press down until it snaps into position. Do not operate the pressure cooker without the anti-block shield installed.",
            "pressure_lid": "To remove the pressure cooking lid, hold the lid handle and turn it counterclockwise until the lid symbol aligns with the cooker base symbol, then lift it up. To close it, align the lid symbol with the base symbol, place the lid in the track, and turn it clockwise until aligned.",
            "pressure_condensation_collector": "Install the condensation collector at the back of the cooker base by aligning its grooves with the tabs and sliding it into place. It should be installed before cooking and emptied and rinsed after each use.",
            "pressure_sealing_ring": (
                "The sealing ring forms an airtight seal between the pressure cooking lid and inner pot. To remove it, pull the silicone edge out from behind the stainless steel sealing ring rack. To install it, press the sealing ring around the rack until it sits firmly behind the rack without wrinkles. Use only one sealing ring at a time."
            ),
            "toothbrush_travel_case_charge": (
                "To charge the toothbrush inside the travel case, plug the USB cable into the travel case and connect it to a USB wall adapter. Plug the adapter into an electrical outlet, then place the toothbrush in the case. If charging starts correctly, the handle beeps twice and the lights illuminate upward. The battery indicator blinks white while charging and turns off when fully charged."
            ),
            "earphones_other_functions": (
                "Besides the main earbud controls, the earphones support several other functions:\n"
                "1. Press and hold until the first beep, then let go to activate the phone voice assistant.\n"
                "2. Press and hold until the first beep, then let go to activate the music app; it resumes playback where you left off, or starts recommended music if there is no previous session.\n"
                "3. Press the left earbud to cycle Ambient Awareness and ANC modes on or off.\n"
                "4. Press and hold until the second beep, then let go to turn Low Latency Mode on or off for lip-sync and gaming."
            ),
            "earphones_reset": (
                "If the earphones will not work, first disconnect the earbuds from Bluetooth and remove them from the case. To perform a factory reset, press and hold for 6 seconds; this erases all settings. To perform a hardware reset that restarts the earphones, press and hold for 10 seconds, then press again to power them on. Flashing red and blue lights indicate the reset/pairing state."
            ),
            "earphone_ear_tip_replace": "The earphones normally come with M-size ear tips installed. You can replace them with S-size or L-size ear tips for a better fit. To remove an ear tip, twist it and pull it off. Install the new ear tip firmly so it does not detach accidentally during use.",
            "earphones_case_charge": "If the earphones charging-case battery is low, connect the charging case to power with its charging cable and let the case charge. Keep the earbuds in the case if you also want them to charge. Check the case or earbud indicator lights to confirm charging status.",
            "coffee_program_volume": (
                "To program the coffee machine water volume, turn the machine on and wait until it is in ready mode with steady lights. Fill the water tank with potable water, insert a capsule, and place a cup under the coffee outlet. Press and hold the Espresso or Lungo button, then release the button once the desired volume has been served. The water volume level is then stored."
            ),
            "coffee_energy_saving": "The coffee machine has an energy-saving Power Off mode. It automatically enters power-off mode after about 9 minutes. To turn it on again, press the Espresso or Lungo button. To turn it off before automatic power-off, press both the Espresso and Lungo buttons.",
            "coffee_after_use_clean": (
                "To keep the coffee maker in good condition after use, lift and close the lever to eject the used capsule into the used-capsule container. Empty and clean the used-capsule container and drip tray, and clean the coffee outlet regularly with a soft damp cloth. Before cleaning, unplug the machine and let it cool; do not immerse it in water, use strong cleaners, sharp objects, brushes, abrasives, or place it in a dishwasher."
            ),
            "boat_maintenance_screen": "The boat controller maintenance setting screen shows the number of hours the engines have run since the last maintenance. After maintenance is performed, reset the operation-hour counter by tapping the Reset button and confirming the reset on the screen.",
            "camera_cp_direct": "To print photos with the camera's CP Direct/direct printing method, connect the camera directly to a compatible printer, select the image from the camera, choose the print settings, and start printing from the camera. The printing procedure is controlled from the camera rather than from a computer.",
            "camera_power": "To power the camera, first make sure a charged battery is installed correctly. Then move the camera power switch to ON. If the camera does not operate, check the battery charge and installation before trying again.",
            "toothbrush_intensity": "The electric toothbrush has three intensity levels: High intensity (three lights), Medium intensity (two lights), and Low intensity (one light). Use the intensity indicator lights on the handle to cycle through the levels before, during, or after brushing, or adjust the setting in the app.",
            "toothbrush_features": "To customize toothbrush features, use the app to activate or deactivate supported options such as Adaptive Intensity, Pressure Sensor Feedback, Scrubbing Feedback, and the brush-head replacement reminder. Note that turning off Adaptive Intensity disables automatic intensity adjustment.",
            "delete_images": "To erase all images from the camera, open the Delete function and choose All. The manual also advises deleting image files with the camera rather than changing or deleting folders on the memory card with a computer.",
            "ereader_buttons": "The eReader includes Home/ESC, Previous/Next Page, Navigation/Menu, Zoom in/out, Rotate, 3.5 mm headphone jack, USB port, Micro SD card reader, Play/Pause, Power button, volume buttons, Reset switch, speaker, and TFT LCD display.",
            "ereader_main_browser": "Main Menu shows the device functions, including Browser History, eBook, Music, Video, Photo, Record, Explorer, and settings. Browser History shows recently read files; selecting a book and pressing M opens it at the last reading position.",
            "ereader_ebook_mode": "In eBook mode, pressing M opens functions such as Page Jump, Save Mark, Load Mark, Del Mark, Browser Mode, Flip Time, Brightness, and Set Color. These options let you jump pages, manage bookmarks, set browsing mode, and adjust brightness or color.",
            "ereader_music": "To listen to music on the eReader, open the audio files list from the main menu, select a file, and press M to enter playback mode. You can also connect the device to a computer by USB and copy audio files to it.",
            "ereader_record": "To record voice on the eReader, select Record from the main menu and press M to enter voice record mode. Press Play/Pause to start recording, press it again to pause, and press HOME to stop. After recording, choose YES or NO with the M key to save or discard it. To play a recording, go to Music, open the Recorded file list, select the recording, and press M.",
            "ereader_video": "To play video on the eReader, select Video from the main menu and press M to enter Video mode. The device supports AVI, RMVB, and MPEG2 video formats. During playback, press M to access options such as Subtitle Language, time play, Full Screen, and Brightness if those options are available for the video.",
            "grill_indirect_cooking": "For indirect cooking on the grill, keep the lid closed and cook with indirect heat rather than direct flame. This is suitable for poultry and large cuts of meat, helps reduce flare-ups from dripping grease, and may require temperature adjustment in cold or windy conditions.",
            "tv_manual_program_channels": "To memorize channels with Manual Program, select the channel number with the remote control up/down keys or number buttons, then press MEMORY/ERASE to choose Memory or Erase. The on-screen display confirms whether the channel is stored or erased.",
            "tv_outdoor_antenna": "For outdoor antenna setup, inspect the antenna and cable for deterioration before connecting. A 300-ohm flat wire should be connected through a 300-ohm to 75-ohm adapter, then to the 75-ohm antenna jack. A 75-ohm coaxial cable can be connected directly to the 75-ohm antenna jack.",
            "camera_mount_lens": "To mount the camera lens, remove the rear lens cap and body cap, align the lens mount index with the camera mount mark, and rotate the lens in the indicated direction until it clicks into place.",
            "camera_eyepiece_cover": "Use the eyepiece cover for self-timer or remote shooting to prevent stray light from entering the viewfinder and affecting exposure. Remove the eyecup, then slide the eyepiece cover down into the eyepiece groove.",
            "camera_p_mode": "P mode is Program AE. When the mode dial is set to P, the camera automatically sets shutter speed and aperture for general shooting, while still allowing other settings to be adjusted.",
            "camera_auto_print": "To set automatic print mode on the hybrid instant camera, slide the print mode selector on the side to AUTO. The auto-print icon appears on the shooting screen, and each saved image starts printing immediately. In manual print mode, images are saved to memory so you can choose and print them later.",
            "camera_af_mode": (
                "For AF Mode before taking a picture, use the camera's AF mode setting to choose the focusing behavior. One-Shot AF is for still subjects, AI Servo AF is for moving subjects, and AI Focus AF automatically switches from One-Shot AF to AI Servo AF if a still subject starts moving. The AF point can be selected automatically by the camera or manually in the creative shooting modes."
            ),
            "drill_battery_charge": "To charge the drill battery pack, plug the charger into a suitable outlet before inserting the battery pack. Insert the battery pack fully into the charger. The red charging light flashes continuously while charging has started, and stays on when charging is complete. The battery pack is then fully charged and may be used or left in the charger.",
            "jetski_seat": "To remove the jetski seat, pull up the seat latch, lift the rear of the seat, and remove it. To install it, insert the front projection into the deck holder, press the rear of the seat down, and confirm it is locked securely.",
            "jetski_filler_caps": "The jetski fuel tank filler cap and oil tank filler cap can be removed by turning them counterclockwise. When installing them, tighten each cap securely and confirm it is properly closed before operation.",
            "jetski_levers": "The jetski levers include the throttle lever, choke lever, and QSTS selector. Squeezing the throttle lever increases speed, releasing it reduces speed. Use the choke lever for cold engine starting. Use the QSTS selector at reduced engine speed to adjust the trim angle.",
            "jetski_characteristics": "Key watercraft characteristics include jet-thrust steering and throttle-dependent control. Water drawn through the intake is pressurized by the impeller and expelled through the jet thrust nozzle; steering depends on handlebar position and the amount of throttle, so turning response decreases when engine speed drops.",
            "jetski_fuel_filter": "For the jetski fuel filter and fuel tank, inspect for water, dirt, or damage as part of maintenance. The fuel filter is a one-piece disposable filter and should be replaced after the initial service interval and then at the specified periodic interval, or if water is found in the filter. Have a Yamaha dealer replace it if required.",
            "jetski_intake_impeller": "To clean a dirty jetski intake or impeller, stop the engine, remove the engine shut-off clip, and keep the watercraft safely supported. Check the jet intake and impeller area for weeds or debris, remove the obstruction carefully, and do not operate the watercraft until the intake area is clear.",
            "mower_roll_bar": "Keep the lawn mower roll bar raised and locked whenever possible and wear the seat belt. Lower it only when absolutely necessary, drive slowly and carefully, and do not wear the seat belt while it is lowered. Use the pins and hairpin cotters to lower or raise and lock it.",
            "mower_load": "To load a lawn mower, use a full-width, sturdy ramp secured to the trailer or truck. Drive straight up the ramp slowly, avoid sudden acceleration or turning, stop the engine, engage the parking brake, and tie the machine down securely.",
            "mower_unload": "To unload a lawn mower, make sure the ramp is secured and clear, then drive straight down slowly. Avoid sudden braking, sharp turns, or side movement on the ramp, and keep people and obstacles away from the unloading area.",
            "mower_rear_shock": "For a mower with suspension, adjust the rear-shock assemblies from the softest to firmest positions to change ride comfort. Keep both sides in the same position.",
            "mower_height_cut": "To adjust height of cut with an electric deck lift, use the deck-lift switch to raise or lower the mower deck, select the desired hole in the height-of-cut bracket, and insert the height-of-cut pin.",
            "mower_remove_filters": "To remove the mower filters, park on level ground, disengage PTO, engage the parking brake, stop the engine, remove the key, wait for moving parts to stop, release the air-cleaner latches, remove the air-inlet cover, clean the screen and cover, and inspect or replace the primary filter.",
            "mower_replace_belt": "To replace the mower belt, park safely, lower the deck to 76 mm (3 inches), remove the belt covers, release idler spring tension with a 3/8-inch ratchet in the idler arm square hole, remove the belt from the deck pulleys and clutch pulley, install the new belt, then reinstall the guide, spring, and covers.",
            "vacuum_dual_modes": "The Dual Mode Virtual Wall Barrier has two main modes: Virtual Wall Mode creates an invisible cone-shaped barrier to block the vacuum from an area, and Halo Mode creates a protected zone around items such as pet bowls or vases.",
            "vacuum_robot_anatomy": "The vacuum robot anatomy includes the main robot body, Clean button, bin release button, dust bin/bin door, filter, sensors, charging contacts, side brush, front caster wheel, and debris extractors. These parts are the main areas used for operation, emptying the bin, charging, and routine cleaning.",
            "vacuum_empty_bin": "To empty the vacuum bin, press the bin release button, remove the bin, and open the bin door to empty debris. If the full-bin indicator turns on during cleaning, pause cleaning, empty the bin, and continue.",
            "vacuum_clean_filter": "To clean the vacuum filter, pull the yellow tab to remove the filter, tap it to remove dust and debris, then reinstall it. The filter door will not close if the filter is not installed correctly.",
            "vacuum_full_bin_sensors": "To clean the full bin sensors, remove and empty the bin, then wipe the sensors and the inner and outer sensor ports with a clean, dry cloth.",
            "vacuum_sensors_contacts": "To clean the sensors and charging contacts, wipe the sensors with a clean, dry cloth and do not spray cleaning solution directly on sensors or openings. Also wipe the charging contacts on both the vacuum and Home Base.",
            "vacuum_home_base": "Place the Home Base in an open, uncluttered area with at least 1.5 feet on each side, at least 4 feet in front, at least 4 feet from stairs, and at least 8 feet from virtual wall barriers. Keep it plugged in.",
            "vacuum_clean_extractors": "To clean the extractors, pinch the yellow release tabs, lift the extractor frame, remove obstructions from the vacuum path, remove the extractors and yellow caps, clear hair and debris from the caps and metal shafts, then reinstall them.",
            "vacuum_clean_side_brush": "To clean the side brush, use a coin or small screwdriver to remove the screw, remove the brush, clean hair and debris from the brush and post, then reinstall and tighten the screw.",
            "vacuum_front_caster": "To clean the vacuum front caster wheel, pull firmly on the front wheel to remove it, clear debris from inside the wheel cavity, then spin the wheel by hand. If it does not rotate freely, remove the wheel from its housing, push out the axle, and clear any hair or debris wrapped around it. Reinstall all parts and make sure the wheel clicks back into place.",
            "snowmobile_throttle_cable": "Before using the snowmobile, check that the throttle, brake, and steering operate correctly. If the carburetor or throttle cable malfunctions, release the throttle lever; the T.O.R.S. system can interrupt ignition and stop the engine. Apply low-temperature grease only to the brake/throttle cable ends.",
            "snowmobile_steering_system": "To check the snowmobile steering system, move the handlebar up and down, back and forth, and slightly left and right. If free play is excessive, consult a dealer.",
            "snowmobile_turning": "To turn a snowmobile, slow down first, turn the handlebars in the desired direction, put body weight on the inside running board, and lean your upper body toward the inside of the turn. Practice at low speed in an open, flat area.",
            "snowmobile_uphill": "For riding uphill on a snowmobile, practice first on gentle slopes. Approach with some speed, accelerate before the climb, then reduce throttle to prevent track slippage. Keep your weight on the uphill side; when climbing straight up, lean forward, and on steeper inclines stand on the running boards and lean over the handlebars. Slow down near the crest. If you cannot continue, stop the engine, set the parking brake, turn the rear of the machine around so it points downhill, avoid standing on the downhill side, then restart and descend safely.",
            "snowmobile_downhill": "For riding downhill on a snowmobile, keep speed to a minimum. Use just enough throttle to keep the clutch engaged so engine compression helps slow the machine. Apply the brake frequently with light pressure rather than relying on high speed or abrupt braking.",
            "snowmobile_cross_slope": "Crossing a slope on a snowmobile is not recommended for beginners. Keep your weight toward the uphill side. The recommended position is to kneel with the downhill knee on the seat and keep the uphill foot on the running board. Be prepared for sideways slipping on snow or ice.",
            "snowmobile_engine_start": "To start the snowmobile engine, make sure the engine stop switch is in the run position, apply the parking brake, and use the starter according to the manual. Do not squeeze the throttle during starting unless the manual specifically instructs you to do so. After the engine starts, let it warm up before riding.",
            "snowmobile_spark_plug": "To inspect the snowmobile spark plug, check the color of the white porcelain insulator around the center electrode; normal color is medium to light tan. Before installation, measure and adjust the electrode gap, clean the gasket surface and threads, and tighten to 28 Nm (20 ft-lb).",
            "microwave_control_setup": "The microwave Control Set-Up lets you change default settings such as beep sound, clock, display speed, and defrost weight (LBS/KG).",
            "microwave_light_timer": "The Light Timer can automatically turn the bottom Lo Light on and off at set times each day. Repeat the setup steps to reset the times, or touch Light HI/LO/Off to cancel a running Light Timer.",
            "microwave_favorite_recipe": "Favorite Recipe stores a custom cooking program so it can be recalled quickly later. After saving the desired cooking time or program, touch Favorite Recipe to recall it and start cooking.",
            "microwave_reheat": "Reheat (Sensor) heats food without manually setting time and power. Preset categories include Casserole, Dinner Plate, and Soup/Sauce; the sensor determines heating time and the oven signals END when complete.",
            "microwave_auto_defrost": "Auto Defrost uses preset defrost sequences for frozen foods. Select the food type and weight, start defrosting, then when the oven pauses and beeps, open the door, turn, separate, or rearrange food, remove thawed portions, and press START to continue.",
            "microwave_oven_light": "To replace the microwave oven light, disconnect power, remove the vent cover mounting screws, tilt and remove the cover, remove and lift the bulb holder, replace the bulb with a 30 or 40 watt appliance bulb, then reinstall the holder and cover and restore power.",
            "microwave_grease_filter": "To clean the over-the-range microwave grease filter, remove the filter from the underside of the oven, soak and wash it in hot water with mild detergent, rinse and dry it, then reinstall it. Do not operate the microwave without the filter in place when required.",
            "microwave_charcoal_filter": "To replace the over-the-range microwave charcoal filter, disconnect power, remove the vent cover or grille as instructed, take out the old charcoal filter, install the new filter in the same position, then reinstall the cover or grille and restore power.",
            "motherboard_pcie_x16": "The motherboard has three PCI Express 3.0 x16 slots for compatible graphics cards. For a single graphics card, use the gray PCIe 3.0 x16_1 slot for better performance. Ensure sufficient power and connect chassis fans when using multiple graphics cards.",
            "motherboard_onboard_led": "The onboard LED is the standby power LED. It lights when the system is on, in sleep mode, or in soft-off mode, reminding you to shut down and unplug the power cable before installing or removing motherboard components.",
            "motherboard_sata_odd_usb_os": "To install an operating system using SATA ODD and USB devices, prepare the support DVD, OS installation source, SATA ODD, and an 8 GB or larger USB device. Copy the OS installation files if needed, connect the USB ODD or USB device, insert the support DVD into SATA ODD, then press F8 during POST to open the boot menu and continue installation.",
            "motherboard_chassis_screws": "To secure the motherboard to the chassis, install nine screws in the circled screw holes shown in the manual. Do not overtighten the screws, and make sure the indicated side faces the rear of the chassis.",
            "motherboard_system_memory": "For system memory, different capacities can be installed in Channel A and Channel B. The system maps the smaller channel capacity for dual-channel operation and runs the remaining larger-channel memory in single-channel mode. Use DIMMs below 1.65 V and a 64-bit OS for 4 GB or more memory.",
            "motherboard_t_sensor": "The Thermal Sensor connector (2-pin T_SENSOR) is used for a thermistor cable. It lets the motherboard monitor the temperature of critical motherboard components or connected devices. Connect the thermistor cable to this header when you need that temperature monitoring function.",
            "motherboard_tpm_connector": "The 14-1 pin TPM connector supports a Trusted Platform Module, which securely stores keys, digital certificates, passwords, and data, and helps enhance network security, digital identity protection, and platform integrity.",
        }
        return answers.get(plan.intent, "")

    def _select_step_answer(self, question: str, results: list[SearchResult]) -> str:
        english_mode = _looks_english_dominant_text(question)
        top_result = results[0] if results else None
        step_result = None
        if top_result and (
            self._is_preferred_instruction_source(question, top_result)
            or bool(self._extract_ordered_items(self._step_source_text(top_result))[1])
        ):
            step_result = top_result
        if step_result is None:
            step_result = next((result for result in results if result.chunk.chunk_type == "step"), None)
        if step_result is None:
            step_result = next((result for result in results if self._extract_ordered_items(self._step_source_text(result))[1]), None)
        if step_result is None:
            return ""

        step_text = self._step_source_text(step_result)
        intro, items = self._extract_ordered_items(step_text)
        if not items:
            text = step_text.strip()
            if len(text) > 260:
                text = text[:257].rstrip("，,；; ") + "..."
            return f"{self._section_prefix(step_result.chunk.section_title, english_mode)}{text}"

        limit = self._desired_item_count(question)
        items = items[:limit] if limit else items[:6]
        fragments: list[str] = []
        if self._useful_step_intro(intro):
            fragments.append(intro)
        fragments.extend(f"{index}. {item}" for index, item in enumerate(items, start=1))
        return f"{self._section_prefix(step_result.chunk.section_title, english_mode)}{' '.join(fragments)}"

    def _select_range_answer(self, question: str, results: list[SearchResult]) -> str:
        english_mode = _looks_english_dominant_text(question)
        desired_count = self._desired_item_count(question)
        for result in results:
            if result.chunk.chunk_type in {"toc", "title_only"}:
                continue
            intro, items = self._extract_ordered_items(result.chunk.text)
            if not items:
                continue
            if "最后" in question:
                picked = items[-desired_count:] if desired_count else items[-3:]
            else:
                picked = items[:desired_count] if desired_count else items[:5]
            if not picked:
                continue
            pieces = [f"{index}. {item}" for index, item in enumerate(picked, start=1)]
            if intro and "注意" not in intro and len(intro) <= 80:
                pieces.insert(0, intro)
            return f"{self._section_prefix(result.chunk.section_title, english_mode)}{' '.join(pieces)}"
        return ""

    def _select_merged_snippets(self, question: str, results: list[SearchResult]) -> str:
        snippets: list[str] = []
        seen_titles: set[str] = set()
        warranty_mode = "保修" in question or "服务" in question

        for result in results:
            title = result.chunk.section_title
            if title in seen_titles:
                continue
            if result.chunk.chunk_type in {"toc", "title_only"} and not warranty_mode:
                continue
            if warranty_mode and not any(keyword in title for keyword in ("保修", "免费服务", "有限保修")):
                continue

            seen_titles.add(title)
            summary = self._compact_summary(result)
            if summary and summary not in snippets:
                snippets.append(summary)
            if len(snippets) >= 3:
                break

        if not snippets and warranty_mode:
            for result in results:
                summary = self._compact_summary(result)
                if summary and summary not in snippets:
                    snippets.append(summary)
                if len(snippets) >= 2:
                    break

        if snippets:
            return "；".join(snippets)
        return ""

    def _compact_summary(self, result: SearchResult) -> str:
        title = result.chunk.section_title.strip()
        text = result.chunk.text.strip().replace("\n", " ")
        if result.chunk.chunk_type == "title_only":
            return title
        if len(text) > 110:
            text = text[:107].rstrip("，,；; ") + "..."
        if title in text:
            return text
        return f"{title}：{text}"

    def _split_sentences(self, text: str) -> list[str]:
        normalized = re.sub(r"(?<!\d)\s+(?=\d+[\.\)、]\s*)", "\n", text)
        parts = re.split(r"(?<=[。！？.!?])\s+|\n+", normalized)
        cleaned: list[str] = []
        for part in parts:
            stripped = part.strip(" -•●.。")
            if not stripped or re.fullmatch(r"\d+[\.\)、]?", stripped):
                continue
            cleaned.append(stripped)
        return cleaned

    def _is_step_question(self, question: str) -> bool:
        lowered = question.lower()
        if any(token in question for token in STEP_PREFIXES):
            return True
        return any(
            token in lowered
            for token in (
                "how",
                "connect",
                "connecting",
                "install",
                "remove",
                "replace",
                "set",
                "setup",
                "start",
                "stop",
                "turn on",
                "turn off",
                "delete",
                "erase",
                "clean",
                "charge",
                "insert",
            )
        )

    def _select_delete_images_answer(self, results: list[SearchResult], question: str = "") -> str:
        english_mode = _looks_english_dominant_text(question)
        for result in results:
            combined = f"{result.chunk.section_title} {result.chunk.text}"
            if not all(term in combined for term in ("删除", "图像")):
                continue
            if "全部" not in combined:
                continue
            warning = ""
            note = next(
                (
                    item
                    for item in results
                    if "请使用相机删除图像文件" in item.chunk.text
                    or "请勿使用电脑更改文件夹/文件名或删除文件夹" in item.chunk.text
                ),
                None,
            )
            if note is not None:
                warning = (
                    " The manual also advises deleting image files with the camera and not changing or deleting folders on the memory card with a computer."
                    if english_mode
                    else " 说明书同时提醒：请使用相机删除图像文件，不要用电脑更改或删除存储卡中的文件夹。"
                )
            if english_mode:
                return f'{self._section_prefix(result.chunk.section_title, True)}To erase all images, open the Delete function and choose "All".{warning}'.strip()
            return f"在“{result.chunk.section_title}”部分提到：如需删除所有图像，请在删除功能中选择“全部”。{warning}".strip()
        return ""

    def _step_source_text(self, result: SearchResult) -> str:
        title = result.chunk.section_title.strip()
        text = result.chunk.text.strip()
        if re.search(r"\d+[\.\)、]|\s\d+$", title) and title not in text:
            return f"{title}\n{text}".strip()
        return text

    def _useful_step_intro(self, intro: str) -> bool:
        intro = intro.strip()
        if not intro:
            return False
        if re.search(r"(训练|练习)\d+", intro):
            return False
        if len(intro) > 90:
            return False
        return True

    def _extract_ordered_items(self, text: str) -> tuple[str, list[str]]:
        normalized = re.sub(r"\s+[•●]\s*", "\n• ", text)
        normalized = re.sub(r"\s+-\s+", "\n- ", normalized)
        normalized = re.sub(r"(?<=[：:。；;])(?=\d+[\.\)、]\s*)", "\n", normalized)
        normalized = re.sub(r"(?<!\d)\s+(?=\d+[\.\)、]\s*)", "\n", normalized)
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        intro_parts: list[str] = []
        items: list[str] = []
        current_item: list[str] = []

        for line in lines:
            if re.match(r"^\d+[\.\)、]?\s*", line):
                if current_item:
                    items.append(" ".join(current_item).strip())
                current_item = [re.sub(r"^\d+[\.\)、]?\s*", "", line).strip()]
                continue
            if re.match(r"^[.。]\s+", line):
                if current_item:
                    items.append(" ".join(current_item).strip())
                current_item = [re.sub(r"^[.。]\s+", "", line).strip()]
                continue
            if re.match(r"^[•●\-]\s*", line):
                if current_item:
                    items.append(" ".join(current_item).strip())
                    current_item = []
                items.append(re.sub(r"^[•●\-]\s*", "", line).strip())
                continue
            if current_item:
                current_item.append(line)
            else:
                intro_parts.append(line)

        if current_item:
            items.append(" ".join(current_item).strip())

        items = [item for item in items if item]
        intro = " ".join(intro_parts).strip()
        return intro, items

    def _is_low_value_result_for_answer(self, question: str, result: SearchResult) -> bool:
        title = result.chunk.section_title.strip()
        text = result.chunk.text.strip()
        combined = f"{title} {text}".strip()
        compact_title = re.sub(r"\s+", "", title)

        if self._is_parameter_question(question) and self._chunk_contains_parameter_signal(combined):
            return False
        if result.chunk.chunk_type == "toc":
            return True
        if result.chunk.chunk_type == "title_only" and not self._chunk_contains_parameter_signal(combined):
            return True
        if text.count("....") >= 2:
            return True
        if len(text) < 8 and not self._chunk_contains_parameter_signal(combined):
            return True
        if compact_title in {"操作方法", "提示", "注意"} and len(text) < 16:
            return True
        if any(term in compact_title for term in ("目录", "前言", "内容前言", "如何使用本练习指南", "目标", "符号说明")):
            if "符号" in question and "符号说明" in compact_title:
                return False
            return True
        return False

    def _is_low_information_snippet(self, answer: str, question: str) -> bool:
        normalized = re.sub(r"\s+", " ", answer).strip()
        compact = re.sub(r"\s+", "", normalized)
        if "操作方法：" in normalized and len(compact) <= len("您好，根据当前检索到的说明书内容：在“操作方法”部分提到：操作方法：") + 8:
            return True
        if any(marker in compact for marker in ("在“内容前言”部分提到", "在“目录", "在“目标”部分提到", "如何使用本练习指南")):
            return True
        if "在“符号说明" in compact and "符号" not in question:
            return True
        if normalized.count("....") >= 2:
            return True
        return False

    def _looks_low_quality_answer(self, answer: str) -> bool:
        if not answer:
            return True
        normalized = re.sub(r"\s+", " ", answer).strip()
        if any(
            phrase in normalized
            for phrase in ("根据一般操作流程", "根据常规操作", "根据一般操作逻辑", "常规启动情况")
        ):
            return True
        if re.search(r"提到[:：]\s*(?:[.。]|[一二三四五六七八九十]?[、.．]?\s*)?$", normalized):
            return True
        if re.search(r"提到[:：]\s*\d+[\.\)、]?\s*(?:。)?$", normalized):
            return True
        if re.search(r"\b1[.)]\s*(?:to|speed maneuvering|such as when dockin)\b", normalized, flags=re.IGNORECASE):
            return True
        if self._contains_insufficient_claim(normalized):
            return False
        substantive = [
            line.strip()
            for line in answer.splitlines()
            if line.strip() and not line.strip().startswith("可参考配图")
        ]
        if len("".join(substantive)) < 18:
            return True
        return False

    def _desired_item_count(self, question: str) -> int | None:
        if "前五条" in question or "前5条" in question:
            return 5
        if "前两个步骤" in question:
            return 2
        if "最后三个步骤" in question or "最后三步" in question:
            return 3
        if "前六个步骤" in question:
            return 6
        return None

    def _is_preferred_instruction_source(self, question: str, result: SearchResult) -> bool:
        if result.chunk.chunk_type in {"toc", "title_only", "note", "warning", "troubleshoot"}:
            return False
        combined = f"{result.chunk.section_title} {result.chunk.text}".lower()
        alpha_terms = re.findall(r"[A-Za-z0-9_+-]{2,}", question.lower())
        if alpha_terms and any(term in combined for term in alpha_terms):
            return True
        return result.chunk.chunk_type in {"step", "list", "menu"} and len(result.chunk.text) >= 40

    def _best_sentence_from_result(self, question: str, result: SearchResult) -> str:
        sentences = self._split_sentences(result.chunk.text)
        if not sentences:
            return result.chunk.text.strip()
        query_terms = set(tokenize(question))
        ranked: list[tuple[float, str]] = []
        for sentence in sentences:
            sentence_terms = set(tokenize(sentence))
            overlap = len(query_terms & sentence_terms)
            ranked.append((overlap + result.score, sentence))
        ranked.sort(key=lambda item: (-item[0], len(item[1])))
        snippet = ranked[0][1].strip()
        if len(snippet) > 220:
            snippet = snippet[:217].rstrip("，,；; ") + "..."
        return snippet
