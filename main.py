import json
import os
from typing import List, Dict, Any, TypedDict, Annotated, Sequence
from typing_extensions import TypedDict
from langchain_openai import AzureChatOpenAI   , ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv


# llm = ChatOpenAI(
#     model="Fanar",
#     openai_api_base="https://api.fanar.qa/v1",
#     openai_api_key="9V6w2YnCYm9fygQBsNXtuS7ZMfzT7LII",
#     temperature=0.7,
# )
key_path = ".env"  # Define the path to your .env file
load_dotenv(dotenv_path=key_path, override=True)


llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_ENGINE_NAME"],
    api_version=os.environ["AZURE_API_VERSION"],
    api_key=os.environ["AZURE_API_KEY"],
    temperature=0.7,
)

arabic_dialogue_styles = {
        "حواري": {
            "host_example": "أحمد: يا أهلاً نور! كيف الحال؟ اممم... قوليلي، شو اللي خلاكِ تدخلي هذا المجال؟ <happy>",
            "guest_example": "نور: أهلاً أحمد! الله يعطيك العافية... يعني بصراحة، هاي قصة طويلة شوي [pause: 2s] بس باختصار، كنت أشوف المشاكل حولي وأقول: ليش ما نحلها بالتقنية؟"
        },
        "تعليمي": {
            "host_example": "أحمد: اممم... نور، ممكن تشرحي لنا بطريقة بسيطة، يعني شلون تشتغل هاي التقنية؟ <happy>",
            "guest_example": "نور: طبعاً أحمد! يعني... اههه كيف أشرح [pause: 2s] تخيل إنك عندك نظام ذكي جداً، بس هذا النظام مش إنسان، هو كمبيوتر! واو صح؟"
        },
        "ترفيهي": {
            "host_example": "أحمد: هاي نور! <happy> قوليلي، إيش أغرب موقف صار معكِ في الشغل؟ يعني شي يضحك؟",
            "guest_example": "نور: ههههه واو أحمد! بصراحة مواقف كثيرة... اممم مرة كنت أجرب البرنامج وفجأة [pause: 2s] خلاص ما عاد يشتغل! قعدت أصرخ: وين راح كودي؟! <surprise>"
        },
        "تحليلي": {
            "host_example": "أحمد: نور، بناءً على الإحصائيات الحديثة، وش رايكِ في التحديات الرئيسية اللي تواجه هذا المجال؟",
            "guest_example": "نور: سؤال ممتاز أحمد... يعني إذا نتكلم بشكل تحليلي، عندنا ثلاث تحديات أساسية [pause: 2s] أولها التقنية، ثانيها التمويل، وثالثها... اممم التقبل المجتمعي"
        }
    }

emotional_tags = {
    "Angry", "Happy", "Sad", "Surprised", "Neutral", "Disappointed",
    "Curious", "Confused", "Concerned"
}

background = (
   "ظاهرة العنوسة تشهد ارتفاعاً ملحوظاً في المجتمعات العربية، حيث تشير الإحصائيات إلى أن 30% من النساء في دول الخليج "
   "و 25% من الرجال تجاوزوا سن الثلاثين دون زواج، مقارنة بـ 15% قبل عقدين. "
   "الأسباب متعددة تشمل ارتفاع تكاليف الزواج والسكن، تغير الأولويات المهنية والتعليمية، صعوبة التوافق بين الشريكين، "
   "وتأثير وسائل التواصل الاجتماعي على توقعات الشباب من الزواج. "
   "العوامل الاقتصادية تلعب دوراً كبيراً حيث تصل تكلفة الزواج في بعض الدول العربية إلى 200 ألف دولار شاملة المهر والحفل والسكن. "
   "التغيرات الاجتماعية مثل دخول المرأة سوق العمل بقوة، السفر للدراسة، وتقبل المجتمع للعزوبية أكثر من الماضي "
   "أدت لتأخير قرارات الزواج. "
   "الحلول المقترحة تتضمن برامج التيسير الحكومية، منصات التعارف المحترمة، ورش التأهيل للزواج، "
   "وحملات توعية لتغيير النظرة المجتمعية حول تكاليف الزواج والعمر المناسب له."
)


style_prompts = {
            "حواري": """
أسلوب حواري:
- ركز على الحوار الطبيعي والتفاعل الشخصي بين المقدم والضيف
- أضف لحظات تداخل وتفاعل عفوي
- استخدم لغة ودية ومألوفة
- اجعل النقاش يبدو كمحادثة بين أصدقاء
- أكثر من الأسئلة الشخصية والتجارب الذاتية""",
            "تعليمي": """
أسلوب تعليمي:
- اهتم بتقديم المعلومات بطريقة منظمة ومفصلة
- استخدم أمثلة توضيحية وتشبيهات مفهومة
- اجعل الضيف يشرح المفاهيم خطوة بخطوة
- أضف أسئلة استيضاحية من المقدم
- ركز على الفهم العميق للموضوع مع الحفاظ على الطبيعية""",
            "ترفيهي": """
أسلوب ترفيهي:
- أضف عناصر مرحة وقصص شخصية طريفة
- استخدم الفكاهة المناسبة والتعليقات الخفيفة
- اجعل النقاش حيوياً ومليئاً بالطاقة
- أضف مواقف مضحكة أو غريبة مرتبطة بالموضوع
- ركز على الجانب الإنساني والممتع من الموضوع""",
            "تحليلي": """
أسلوب تحليلي:
- ركز على التحليل العميق والنقاش المتخصص
- استخدم بيانات وإحصائيات ومراجع علمية
- اطرح أسئلة تحليلية معقدة
- ناقش التحديات والحلول بتفصيل
- اجعل النقاش فكرياً ومتعمقاً مع الحفاظ على الوضوح"""
        }

complexity_guidance = {
    "بسيط": """
    - استخدم تشبيهات من الحياة اليومية
    - اشرح كل مصطلح تقني فوراً
    - اطرح أسئلة تأكيدية: "واضح؟ مفهوم؟"
    - استخدم "يعني" كثيراً للتوضيح
    """,
    "متوسط": """
    - امزج التشبيهات البسيطة مع التقنية
    - اشرح المصطلحات مع إعطاء سياق
    - أضف تفاصيل تدريجياً
    - استخدم "بمعنى آخر" للتوضيح
    """,
    "معقد": """
    - استخدم مصطلحات متخصصة مع سياق
    - أضف أرقام وإحصائيات
    - ناقش التحديات التقنية
    - استخدم "من ناحية تقنية" للتعمق
    """
}


clutural_prompt =  f"""
السياق الثقافي العربي المرتبط بأي موضوع:

ملاحظة مهمة: الأمثلة التالية هي للإرشاد فقط. يجب تطبيق نفس المبادئ على الموضوع الفعلي وليس استخدام الأمثلة حرفياً.

1. أمثال وحكم مرتبطة بالموضوع:
   - ابحث عن أمثال عربية ترتبط مباشرة بطبيعة الموضوع المطروح
   - مقولات مأثورة تتعلق بالمجال (مثل: العلم، التطور، الحكمة، الصبر، العمل)
   - حكم تتناسب مع روح الموضوع سواء كان تقني، ديني، سياسي، أو اجتماعي

2. مراجع إقليمية ومعاصرة:
   - تجارب المنطقة العربية أو الدول العربية في هذا المجال
   - مبادرات محلية أو خليجية أو عربية ذات صلة
   - شخصيات أو مؤسسات عربية رائدة في المجال
   - أحداث تاريخية أو معاصرة مرتبطة بالموضوع

3. التجارب المشتركة:
   - كيف يتفاعل المجتمع العربي مع هذا النوع من المواضيع
   - تحديات مشتركة نواجهها في المنطقة العربية
   - فرص مستقبلية للعالم العربي في هذا المجال
   - نقاط اهتمام مشتركة للجمهور العربي

4. الربط بالقيم العربية والإسلامية:
   - كيف يتماشى الموضوع مع القيم العربية والإسلامية (إن أمكن)
   - الفوائد المجتمعية من منظور ثقافي عربي
   - التحديات الأخلاقية أو الثقافية من منظور عربي
   - الموازنة بين التطور والحفاظ على الهوية
"""

emotional_arc = f"""
القوس العاطفي للحلقة (قابل للتطبيق على أي موضوع):

ملاحظة: هذا إطار عام يجب تطبيقه على الموضوع الفعلي وليس استخدام الأمثلة حرفياً.

1. البداية (حماس وترقب):
   - المقدم يبدأ بحماس طبيعي حسب شخصيته وعلاقته بالموضوع
   - تشويق للموضوع يناسب خلفية المستمعين العرب
   - ترقب لما سيشاركه الضيف من خبرات

2. الوسط (فضول وتعلم):
   - فضول متزايد من المقدم حول جوانب الموضوع
   - لحظات دهشة عند اكتشاف معلومات جديدة أو غير متوقعة
   - تفاعل إيجابي مع شروحات وتحليلات الضيف

3. التحدي (قلق أو تساؤل):
   - طرح تحديات أو مخاوف مشروعة حول الموضوع
   - نقاش حول الصعوبات أو المقاومة أو الجدل المحيط بالموضوع
   - تساؤلات أخلاقية أو ثقافية أو عملية حسب طبيعة الموضوع

4. الحل والفهم:
   - وصول لفهم أعمق أو منظور جديد مع الضيف
   - إيجاد حلول أو إجابات للتساؤلات المطروحة
   - تفاؤل حذر أو واقعي حسب طبيعة الموضوع

5. النهاية (إلهام ودعوة للعمل):
   - رسالة ملهمة تناسب الجمهور العربي وطبيعة الموضوع
   - خطوات عملية يمكن للمستمعين اتخاذها (إن أمكن)
   - تطلع للمستقبل والدور العربي في هذا المجال
"""


class PodState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    enhanced_outline: Annotated[list, add_messages]
    main_dialogue: Annotated[list, add_messages]
    topic_analysis: Dict[str, Any]
    topic: str
    style: str
    information: str
    host_persona: str
    guest_persona: str
    outline: str
    main_script: str
    complexity: str


def classify_topic(state: PodState) -> PodState:
    prompt = f"""
أنت خبير في تحليل المواضيع وتصنيفها لإنتاج بودكاست عربي طبيعي وجذاب.

المهمة: حلل الموضوع التالي وحدد أفضل نهج لتقديمه في بودكاست عربي.

الموضوع: {state['topic']}
المعلومات الأساسية: {state['information']}

قم بتحليل الموضوع وإرجاع النتيجة بصيغة JSON تحتوي على:

{{
    "primary_category": "الفئة الرئيسية",
    "category_justification": "سبب اختيار هذه الفئة بناءً على طبيعة الموضوع",
    "secondary_tags": ["تصنيفات فرعية مرتبطة"],
    "optimal_style": "الأسلوب الأمثل للمناقشة",
    "emotional_intent": "القصد العاطفي للحلقة",
    "discourse_pattern": "نمط الخطاب المناسب",
    "audience_engagement_goal": "هدف تفاعل الجمهور",
    "cultural_sensitivity_level": "مستوى الحساسية الثقافية",
    "controversy_potential": "احتمالية الجدل",
    "key_discussion_angles": [
        "زوايا النقاش الرئيسية المتوقعة",
        "النقاط التي ستثير اهتمام الجمهور العربي"
    ],
    "natural_tension_points": [
        "نقاط التوتر الطبيعية في الموضوع",
        "الجوانب التي قد تثير جدلاً صحياً"
    ],
    "cultural_connection_opportunities": [
        "فرص الربط بالثقافة العربية",
        "المراجع المحلية والإقليمية ذات الصلة"
    ]
}}

الفئات المتاحة:
1. "العلوم والتكنولوجيا" - للمواضيع التقنية والعلمية والابتكارات
2. "السياسة والشؤون العامة" - للمواضيع السياسية والأحداث الجارية والقضايا العامة
3. "القضايا الاجتماعية" - للمواضيع المجتمعية والعلاقات والقيم والتحديات الاجتماعية
4. "الرياضة والترفيه" - للمواضيع الرياضية والفنية والترفيهية
5. "التاريخ والثقافة" - للمواضيع التاريخية والتراثية والثقافية

الأساليب المتاحة:
- "حواري" - حوار طبيعي وودي بين المقدم والضيف
- "تعليمي" - تركيز على الشرح والتعليم بطريقة ممتعة
- "ترفيهي" - مرح وخفيف مع لمسات فكاهية
- "تحليلي" - نقاش عميق ومتخصص وتحليلي

أنماط الخطاب:
- "رسمي" - لغة رسمية ومحترمة
- "ودي" - لغة دافئة ومألوفة
- "جدلي" - نقاش حيوي مع وجهات نظر متعددة
- "سردي" - أسلوب حكواتي وقصصي

القصد العاطفي:
- "تعليمي" - إكساب المعرفة والفهم
- "ملهم" - تحفيز وإلهام المستمعين
- "ترفيهي" - إمتاع وتسلية
- "مثير للجدل" - إثارة النقاش والتفكير النقدي
- "مطمئن" - تهدئة المخاوف وتقديم الراحة النفسية

مستوى الحساسية الثقافية:
- "عالي" - يتطلب حذراً شديداً في التعامل
- "متوسط" - يحتاج مراعاة ثقافية معتدلة  
- "منخفض" - موضوع مقبول عموماً

احتمالية الجدل:
- "عالية" - موضوع مثير للجدل بطبيعته
- "متوسطة" - قد يثير بعض الاختلافات
- "منخفضة" - موضوع مقبول عموماً

تعليمات مهمة:
- حلل الموضوع بعمق وليس بشكل سطحي
- اعتبر السياق الثقافي العربي في التحليل
- ركز على ما يجعل الموضوع جذاباً للجمهور العربي
- تأكد أن التصنيف يخدم إنتاج محتوى طبيعي وتلقائي
- لا تضع علامات ```json في البداية أو النهاية
- أرجع JSON صحيح فقط بدون أي نص إضافي
-المدة 10 دقائق هي المدة المثلى للحلقة
"""
    

    response = llm.invoke(prompt)
    state['topic_analysis'] = json.loads(response.content)
    return state


def persona_gen(state: PodState) -> PodState:
    classification = state['topic_analysis']
    print(classification)
    prompt = f"""
أنت خبير في تصميم شخصيات البودكاست العربي.

المهمة: أنشئ مقدم وضيف بسيطين ومناسبين لهذا الموضوع.

الموضوع: {state['topic']}
المعلومات: {state['information']}
الفئة: {classification['primary_category']}
الأسلوب المطلوب: {classification['optimal_style']}

أرجع النتيجة بصيغة JSON بسيط:

{{
    "host": {{
        "name": "اسم المقدم",
        "age": عمر رقمي,
        "background": "خلفية مختصرة في جملة واحدة",
        "personality": "وصف شخصيته في جملة واحدة",
        "speaking_style": "أسلوب حديثه في جملة واحدة"
    }},
    "guest": {{
        "name": "اسم الضيف", 
        "age": عمر رقمي,
        "background": "خلفية مختصرة في جملة واحدة",
        "expertise": "مجال خبرته في جملة واحدة",
        "personality": "وصف شخصيته في جملة واحدة",
        "speaking_style": "أسلوب حديثه في جملة واحدة"
    }},
    "why_good_match": "لماذا هذا المقدم والضيف مناسبان لهذا الموضوع - جملة واحدة"
}}

متطلبات:
- أسماء عربية مألوفة
- شخصيات بسيطة وقابلة للتصديق
- مناسبة للموضوع والأسلوب المطلوب
- المقدم فضولي والضيف خبير أو صاحب تجربة
- لا تضع علامات ```json
- أرجع JSON فقط
"""
    response = llm.invoke(prompt)
    state['host_persona'] = response.content['host']
    state['guest_persona'] = response.content['guest']
    return state

def base_outline_gen(state: PodState) -> PodState:
    primary_category = state['topic_analysis'].get("primary_category", "")
    optimal_style = state['topic_analysis'].get("optimal_style", "")
    discourse_pattern = state['topic_analysis'].get("discourse_pattern", "")
    key_discussion_angles = state['topic_analysis'].get("key_discussion_angles", [])
    natural_tension_points = state['topic_analysis'].get("natural_tension_points", [])
    recommended_duration = state['topic_analysis'].get("recommended_duration", "10 دقيقة")
    host_name = state['host_persona'].get("name", "المقدم")
    guest_name = state['guest_persona'].get("name", "الضيف")
    host_background = state['host_persona'].get("background", "")
    guest_expertise = state['guest_persona'].get("expertise", "")
    global background
    style_guidance = style_prompts.get(state['style'], "")
    prompt = f"""
أنت خبير في تصميم هيكل المحتوى للبودكاست العربي.

المهمة: أنشئ مخطط محتوى منطقي ومترابط لهذه الحلقة.

الموضوع: {state['topic']}
المعلومات: {background}

السياق:
- الفئة: {primary_category}
- الأسلوب: {optimal_style}  
- نمط الخطاب: {discourse_pattern}
- المدة المقترحة: {recommended_duration}

الشخصيات:
- المقدم: {host_name} - {host_background}
- الضيف: {guest_name} - {guest_expertise}

زوايا النقاش المهمة: {key_discussion_angles}
نقاط التوتر الطبيعية: {natural_tension_points}

أرجع مخطط المحتوى بصيغة JSON:

{{
    "episode_info": {{
        "title": "عنوان الحلقة",
        "main_theme": "الموضوع الرئيسي في جملة واحدة",
        "target_duration": "المدة المتوقعة",
        "content_style": "نوع المحتوى المطلوب"
    }},
    "content_structure": {{
        "opening": {{
            "hook": "جملة افتتاحية جذابة محددة",
            "topic_introduction": "تقديم الموضوع بوضوح",
            "guest_introduction": "تقديم الضيف مع ربطه بالموضوع",
            "episode_roadmap": "خريطة ما سنناقشه في الحلقة"
        }},
        "main_sections": [
            {{
                "section_title": "عنوان القسم الأول",
                "main_points": [
                    "النقطة الأساسية الأولى",
                    "النقطة الأساسية الثانية", 
                    "النقطة الأساسية الثالثة"
                ],
                "key_questions": [
                    "سؤال رئيسي سيطرحه المقدم",
                    "سؤال متابعة مهم"
                ],
                "expected_insights": [
                    "رؤية متوقعة من الضيف",
                    "معلومة جديدة محتملة"
                ],
                "transition_to_next": "كيف الانتقال للقسم التالي"
            }},
            {{
                "section_title": "عنوان القسم الثاني", 
                "main_points": [
                    "النقطة الأساسية الأولى",
                    "النقطة الأساسية الثانية",
                    "النقطة الأساسية الثالثة"
                ],
                "key_questions": [
                    "سؤال رئيسي للقسم الثاني",
                    "سؤال استكشافي"
                ],
                "expected_insights": [
                    "رؤية متوقعة مختلفة",
                    "تحليل عميق"
                ],
                "transition_to_next": "كيف الانتقال للقسم التالي"
            }},
            {{
                "section_title": "عنوان القسم الثالث",
                "main_points": [
                    "النقطة الأساسية الأولى", 
                    "النقطة الأساسية الثانية",
                    "النقطة الأساسية الثالثة"
                ],
                "key_questions": [
                    "سؤال للقسم الثالث",
                    "سؤال تطبيقي"
                ],
                "expected_insights": [
                    "حلول أو توصيات",
                    "نصائح عملية"
                ],
                "transition_to_next": "كيف الانتقال للختام"
            }}
        ],
        "closing": {{
            "key_takeaways": [
                "الخلاصة الأولى المهمة",
                "الخلاصة الثانية المهمة",
                "الخلاصة الثالثة المهمة"
            ],
            "final_message": "الرسالة الختامية الأساسية",
            "call_to_action": "دعوة للعمل أو التفكير",
            "next_steps": "خطوات يمكن للمستمع اتخاذها",
            "closing_gratitude": "شكر الضيف والمستمعين"
        }}
    }},
    "content_flow": {{
        "logical_progression": "كيف تتدفق الأفكار منطقياً",
        "narrative_arc": "القوس السردي للحلقة",
        "engagement_points": [
            "نقاط جذب انتباه المستمع",
            "لحظات تفاعل متوقعة"
        ],
        "complexity_progression": "كيف تتطور صعوبة المحتوى"
    }},
    "cultural_integration": {{
        "relevant_examples": [
            "أمثلة محلية مناسبة للموضوع",
            "حالات عربية ذات صلة"
        ],
        "cultural_sensitivity_notes": [
            "نقاط حساسة يجب مراعاتها",
            "توجيهات ثقافية"
        ],
        "regional_relevance": "كيف يرتبط الموضوع بالمنطقة العربية"
    }}
}}

تأكد من أن المخطط يشجع على:
1. الحوار الطبيعي والتلقائي
2. إدماج شخصيات الأفراد بشكل أصيل
3. استخدام اللغة العربية الفصحى بطريقة طبيعية
4. إضافة لمسات ثقافية مناسبة
5. خلق لحظات عاطفية حقيقية

مهم جداً - متطلبات المحتوى:
- لا تكتب أوصافاً عامة مثل "ترحيب عام" أو "تقديم الموضوع"
- اكتب المحتوى الفعلي والنصوص المحددة التي سيقولها المقدم والضيف
- كل قيمة يجب أن تحتوي على نص حقيقي وليس وصفاً لما يجب فعله
- استخدم أسماء الشخصيات الفعلية (أحمد، نور) في النصوص
- اجعل كل محتوى مرتبطاً مباشرة بموضوع السيارات ذاتية القيادة
- أرجع المخطط بصيغة JSON صحيحة فقط
- لا تضع علامات ```json في البداية أو ``` في النهاية
- لا تكتب أي نص إضافي قبل أو بعد JSON

مثال على ما هو مطلوب:
بدلاً من: "ترحيب عام بالمستمعين"
اكتب: "أهلاً وسهلاً بكم في حلقة جديدة من بودكاست التقنية والمستقبل، أنا أحمد المصري معكم اليوم"

بدلاً من: "تقديم الموضوع"

اكتب: "اليوم سنتحدث عن السيارات التي تقود نفسها، هذه التقنية المذهلة التي قد تغير حياتنا كلياً"
مهم جداً - متطلبات المحتوى:
- لا تكتب أوصافاً عامة مثل "ترحيب عام" أو "تقديم الموضوع"
- اكتب المحتوى الفعلي والنصوص المحددة التي سيقولها المقدم والضيف
- كل قيمة يجب أن تحتوي على نص حقيقي وليس وصفاً لما يجب فعله
- استخدم أسماء الشخصيات الفعلية (أحمد، نور) في النصوص
- اجعل كل محتوى مرتبطاً مباشرة بموضوع {state['topic']}
- أرجع المخطط بصيغة JSON صحيحة فقط
- لا تضع علامات json في البداية أو  في النهاية
- لا تكتب أي نص إضافي قبل أو بعد JSON

مثال على ما هو مطلوب:
بدلاً من: "ترحيب عام بالمستمعين"
اكتب: "أهلاً وسهلاً بكم في حلقة جديدة من بودكاست التقنية والمستقبل، أنا أحمد المصري معكم اليوم"

بدلاً من: "تقديم الموضوع"
اكتب: "اليوم سنتحدث عن {state['topic']}، هذا الموضوع المذهل الذي قد يغير حياتنا كلياً"

"""
    response = llm.invoke(prompt)

    state['outline'] = response.content
    return state


def outline_enhance_style(state: PodState) -> PodState:
    primary_category = state["topic_analysis"].get("primary_category", "")
    optimal_style = state["topic_analysis"].get("optimal_style", "")
    prompt = f"""
أنت خبير في تطبيق أساليب مختلفة للبودكاست العربي.

المهمة: طبق أسلوب الفئة على المخطط الموجود.

الفئة: {primary_category}
الأسلوب: {optimal_style}

المخطط الحالي:
{state['outline']}

إرشادات حسب الفئة:
- العلوم والتكنولوجيا: أضف شرح مبسط وأمثلة عملية
- السياسة والشؤون العامة: أضف وجهات نظر متعددة ونقاش متوازن
- القضايا الاجتماعية: أضف قصص شخصية وتجارب حقيقية
- الرياضة والترفيه: أضف حماس وذكريات مثيرة
- التاريخ والثقافة: أضف أسلوب قصصي وربط بالحاضر

أرجع النتيجة بصيغة JSON:

{{
    "style_adjustments": {{
        "opening_style": "كيف تبدأ الحلقة بأسلوب الفئة",
        "question_style": "نوع الأسئلة المناسبة للفئة",
        "discussion_approach": "كيف نناقش بأسلوب الفئة",
        "closing_style": "كيف نختتم بأسلوب الفئة"
    }},
    "enhanced_sections": [
        {{
            "title": "عنوان القسم الأول محسن",
            "approach": "كيف نتناول هذا القسم بأسلوب الفئة",
            "key_questions": [
                "سؤال مناسب لأسلوب الفئة",
                "سؤال متابعة"
            ]
        }},
        {{
            "title": "عنوان القسم الثاني محسن",
            "approach": "منهج الفئة لهذا القسم",
            "key_questions": [
                "سؤال للقسم الثاني",
                "سؤال تحليلي"
            ]
        }},
        {{
            "title": "عنوان القسم الثالث محسن",
            "approach": "نهج الفئة للختام",
            "key_questions": [
                "سؤال ختامي",
                "سؤال للمستقبل"
            ]
        }}
    ]
}}

متطلبات:
- طبق أسلوب الفئة {primary_category} فقط
- اجعل التحسينات بسيطة ومباشرة
- ركز على الأسئلة والنهج وليس التفاصيل المعقدة
- لا تضع علامات ```json
- أرجع JSON فقط
"""
    
    response = llm.invoke(prompt)
    state['enhanced_outline'] = response.content
    return state

def outline_enhance_spontanity(state: PodState) -> PodState:
    primary_category = state["topic_analysis"].get("primary_category", "")
    cultural_sensitivity = state["topic_analysis"].get("cultural_sensitivity_level", "")
    host_name = state["host_persona"].get("name", "المقدم")
    guest_name = state["guest_persona"].get("name", "الضيف")
    host_personality = state["host_persona"].get("personality", "")
    guest_personality = state["guest_persona"].get("personality", "")
    prompt = f"""

أنت خبير في إضافة الحيوية والأصالة الثقافية للبودكاست العربي.

المهمة: أضف عناصر التلقائية والثقافة العربية للمخطط النهائي.

الموضوع: {state['topic']}
المعلومات: {state['information']}

السياق:
- الفئة: {primary_category}
- مستوى الحساسية الثقافية: {cultural_sensitivity}
- المقدم: {host_name} - {host_personality}
- الضيف: {guest_name} - {guest_personality}

المخطط المحسن:
{state['enhanced_outline']}

أضف العناصر التالية وأرجع النتيجة بصيغة JSON:

{{
    "final_outline": {{
        "opening_with_spontaneity": {{
            "natural_greeting": "ترحيب طبيعي يناسب شخصية المقدم",
            "spontaneous_hook": "بداية تلقائية جذابة",
            "cultural_opening": "افتتاحية تحمل لمسة ثقافية عربية",
            "personality_touch": "لمسة شخصية من المقدم"
        }},
        "enhanced_sections_with_culture": [
            {{
                "section_title": "عنوان القسم الأول نهائي",
                "natural_transitions": [
                    "انتقال طبيعي لبداية القسم",
                    "ربط تلقائي بالقسم السابق"
                ],
                "spontaneous_moments": [
                    "لحظة تفاعل تلقائي متوقعة",
                    "موقف قد يثير ضحكة أو دهشة"
                ],
                "cultural_references": [
                    "مثل أو حكمة عربية مناسبة",
                    "مرجع ثقافي محلي ذو صلة"
                ],
                "personality_interactions": [
                    "كيف سيتفاعل المقدم بشخصيته",
                    "رد فعل متوقع من الضيف"
                ]
            }},
            {{
                "section_title": "عنوان القسم الثاني نهائي",
                "natural_transitions": [
                    "انتقال للقسم الثاني",
                    "ربط منطقي"
                ],
                "spontaneous_moments": [
                    "لحظة تفاعل للقسم الثاني",
                    "موقف عفوي محتمل"
                ],
                "cultural_references": [
                    "مرجع ثقافي للقسم الثاني",
                    "تجربة عربية مشتركة"
                ],
                "personality_interactions": [
                    "تفاعل شخصيات في القسم الثاني",
                    "ديناميكية متوقعة"
                ]
            }},
            {{
                "section_title": "عنوان القسم الثالث نهائي",
                "natural_transitions": [
                    "انتقال للختام",
                    "تمهيد للخلاصة"
                ],
                "spontaneous_moments": [
                    "لحظة ختامية مؤثرة",
                    "تفاعل نهائي طبيعي"
                ],
                "cultural_references": [
                    "حكمة ختامية عربية",
                    "قول مأثور مناسب"
                ],
                "personality_interactions": [
                    "تفاعل ختامي للشخصيات",
                    "لمسة إنسانية أخيرة"
                ]
            }}
        ],
        "natural_closing": {{
            "spontaneous_conclusion": "ختام تلقائي وطبيعي",
            "cultural_farewell": "وداع يحمل لمسة ثقافية عربية",
            "personality_goodbye": "وداع شخصي من المقدم والضيف",
            "audience_connection": "ربط أخير مع الجمهور"
        }}
    }},
    "spontaneity_guide": {{
        "natural_fillers": [
            "حشو طبيعي مناسب للمقدم",
            "تعبيرات تلقائية للضيف"
        ],
        "reaction_triggers": [
            "ما قد يثير رد فعل طبيعي من المقدم",
            "ما قد يفاجئ الضيف ويجعله يتفاعل"
        ],
        "interruption_points": [
            "نقاط مقاطعة طبيعية محتملة",
            "لحظات تداخل إيجابي"
        ],
        "emotional_moments": [
            "لحظات قد تثير المشاعر",
            "نقاط تأثر طبيعية"
        ]
    }},
    "cultural_authenticity": {{
        "regional_touches": [
            "لمسات إقليمية مناسبة للموضوع",
            "مراجع محلية ذات صلة"
        ],
        "shared_experiences": [
            "تجارب عربية مشتركة",
            "ذكريات جماعية"
        ],
        "language_nuances": [
            "تنويعات لغوية خفيفة",
            "تعبيرات محلية مناسبة"
        ],
        "cultural_values": [
            "قيم عربية مرتبطة بالموضوع",
            "مبادئ ثقافية ذات صلة"
        ]
    }}
}}

متطلبات مهمة:
- كل العناصر يجب أن تكون مرتبطة بالموضوع: {state['topic']}
- مناسبة لمستوى الحساسية الثقافية: {cultural_sensitivity}
- تعكس شخصيات المقدم والضيف المحددة
- طبيعية وغير مفتعلة
- أصيلة ثقافياً للجمهور العربي
- لا تضع علامات ```json
- أرجع JSON فقط
"""
    response = llm.invoke(prompt)
    state['enhanced_outline'] = response.content
    return state
def intro_gen(state: PodState) -> PodState:
    opening_data = state['enhanced_outline']["opening_with_spontaneity"]
    spontaneity = state['enhanced_outline']["spontaneity_guide"]
    cultural = state['enhanced_outline']["cultural_authenticity"]
    prompt = f"""
أنت كاتب محتوى متخصص في البودكاست العربي. مهمتك كتابة نص افتتاحي طبيعي وعفوي لبودكاست عربي.
            
            المتطلبات:
            1. استخدم اللغة العربية الفصحى المبسطة
            2. اجعل الحوار طبيعياً وعفوياً
            3. أدمج العناصر الثقافية العربية
            4. اكتب للمضيف (أحمد الشامي) والضيفة (د. ليلى السعدي)
            5. اجعل النص يبدو كمحادثة حقيقية وليس مكتوباً
            
            الهيكل المطلوب:
            - ترحيب طبيعي من أحمد
            - تقديم الموضوع بطريقة مشوقة
            - تقديم الضيفة
            - بداية تفاعلية

            اكتب نص افتتاحي للبودكاست بناءً على:
            البيانات الافتتاحية:
            {opening_data}
            دليل العفوية:
            {spontaneity}
            العناصر الثقافية:
            {cultural}

        متطلبات الحوار:
    - تأكد أن الحوار يدور حول موضوع: {state['topic']}
    - استخدم أسماء الشخصيات الفعلية في الحوار
    - اجعل كل شخصية تتحدث بأسلوبها المميز
    - أضف تفاعلات طبيعية: <happy>, <surprise>, <overlap> (فقط عند الحاجة)
    - استخدم 70% فصحى و 30% لمسات خليجية
    - أدرج مراجع ثقافية مرتبطة بالموضوع
    - اجعل الحوار يتكيف مع الموضوع
    - أضف فواصل طبيعية [pause: 2s] عند الحاجة
    - ابدأ كل جملة بـ المقدم: أو الضيف:
    - تأكد من الانتقال الطبيعي من السياق السابق
    """
    response = llm.invoke(prompt)
    state['intro'] = response.content
    return state



def script_gen(state: PodState) -> PodState:
    original_intro = state['intro']
    sections = state['enhanced_outline']["enhanced_sections_with_culture"]
    spontaneity = state['enhanced_outline']["spontaneity_guide"]
    dialogue = []
    style_example = arabic_dialogue_styles.get(state['style'], {})
    count = 0
    for section in sections:
        if count == 0:
            previous_section = original_intro
        else:
            previous_section = dialogue[count-1]
        count += 1
        prompt = f"""
    هذا مثال على أسلوب الحوار المطلوب:
    {style_example}

    تذكر أن موضوع الحلقة هو: {state['topic']}
    يجب أن يكون كل الحوار مرتبطاً بهذا الموضوع تحديداً وليس أي موضوع آخر.
        السياق من القسم السابق:
    {previous_section}
    حول القسم التالي إلى حوار تلقائي وطبيعي بين المقدم والضيف:

    المقدم: {state['host_persona']}
    - أسلوب الكلام: يجب أن يظهر في كل جملة
    - التشبيهات المفضلة: مرتبطة بخلفيته
    - ردود أفعاله المتوقعة: حسب شخصيته

    الضيف: {state['guest_persona']}
    - نفس المعايير للضيف
    - كيف يشرح المفاهيم المعقدة حسب شخصيته
    - متى يتحمس ومتى يتردد

    استخدم حشو المحادثة الطبيعي بكثافة متوسطة:
    - تفكير: اممم، اههه، يعني كيف أقول، خلاص
    - تأكيد: طبعاً، تماماً، بالضبط، صحيح
    - تردد: بعنييييي، يعني، اه ما أدري، شوف
    - انفعال: واو، يا الله، ما شاء الله، الله يعطيك العافية
    - ربط: بس، لكن، وبعدين، يا أخي، اسمع
    - خليجي خفيف: شلون، وش رايك، زين، ماشي الحال

    القسم: {section['section_title']}
    المحتوى: {section['section_content']}
    استراتيجية وضع الحشو الطبيعي:
- عند التفكير: "اممم... كيف أشرح هذا"
- عند التردد: "يعني... اههه... مش متأكد"
- عند الحماس: "واو! طبعاً! بالضبط!"
- عند التأكيد: "تماماً، صحيح، بالزبط"
- عند الانتقال: "يا أخي، شوف، اسمع"
- لمسات خليجية عند: الترحيب، التقدير، التعجب
    الأسلوب المطلوب: {state['style']}
    {style_prompts.get(state['style'], '')}

    مستوى التعقيد: {state['complexity']}
    {complexity_guidance.get(state['complexity'], '')}

    استراتيجية وضع الحشو الطبيعي:
- عند التفكير: "اممم... كيف أشرح هذا"
- عند التردد: "يعني... اههه... مش متأكد"
- عند الحماس: "واو! طبعاً! بالضبط!"
- عند التأكيد: "تماماً، صحيح، بالزبط"
- عند الانتقال: "يا أخي، شوف، اسمع"
- لمسات خليجية عند: الترحيب، التقدير، التعجب

    متطلبات الحوار:
    - تأكد أن الحوار يدور حول موضوع: {state['topic']}
    - استخدم أسماء الشخصيات الفعلية في الحوار
    - اجعل كل شخصية تتحدث بأسلوبها المميز
    - أضف تفاعلات طبيعية: <happy>, <surprise>, <overlap> (فقط عند الحاجة)
    - استخدم 70% فصحى و 30% لمسات خليجية
    - أدرج مراجع ثقافية مرتبطة بالموضوع
    - اجعل الحوار يتكيف مع الموضوع
    - أضف فواصل طبيعية [pause: 2s] عند الحاجة
    - ابدأ كل جملة بـ المقدم: أو الضيف:
    - تأكد من الانتقال الطبيعي من السياق السابق

    {section['natural_transitions']}
            اللحظات العفوية:
            {section['spontaneous_moments']}
            المراجع الثقافية:
            {section['cultural_references']}
            التفاعلات الشخصية:
            {section['personality_interactions']}
            دليل العفوية:
            {spontaneity}
            التفاعلات الشخصية:
            {section['personality_interactions']}
"""
        response = llm.invoke(prompt)
        new_dialogue = response.content
        dialogue.append(new_dialogue)

    state['main_script'] = dialogue
    return state

def outro_gen(state: PodState) -> PodState:
    closing_data = state['enhanced_outline']["closing"]
    spontaneity = state['enhanced_outline']["spontaneity_guide"]
    cultural = state['enhanced_outline']["cultural_authenticity"]
    intro_script = state['intro']
    main_script = state['main_script']
    script_so_far = intro_script + "\n\n" + "\n\n".join(main_script)
    prompt = f"""
أنت كاتب محتوى متخصص في البودكاست العربي. مهمتك كتابة خاتمة طبيعية ومؤثرة للبودكاست.
النص المكتوب حتى الآن:
{script_so_far}
            المتطلبات:
            1. اكتب خاتمة تلخص النقاط الرئيسية
            2. استخدم نبرة إيجابية وملهمة
            3. أدمج العناصر الثقافية العربية
            4. اجعل التفاعل طبيعياً بين المضيف والضيفة
            5. أضف دعوة للتفاعل مع الجمهور
            
            الهيكل:
            - تلخيص سريع
            - رسالة إيجابية
            - شكر للضيفة
            - دعوة للتفاعل
            - وداع طبيعي
            اكتب خاتمة للبودكاست بناءً على:
            بيانات الخاتمة:
            {closing_data}
            اللحظات العاطفية:
            {spontaneity["emotional_moments"]}
            القيم الثقافية:
            {cultural["cultural_values"]}

    متطلبات الحوار:
    - تأكد أن الحوار يدور حول موضوع: {state['topic']}
    - استخدم أسماء الشخصيات الفعلية في الحوار
    - اجعل كل شخصية تتحدث بأسلوبها المميز
    - أضف تفاعلات طبيعية: <happy>, <surprise>, <overlap> (فقط عند الحاجة)
    - استخدم 70% فصحى و 30% لمسات خليجية
    - أدرج مراجع ثقافية مرتبطة بالموضوع
    - اجعل الحوار يتكيف مع الموضوع
    - أضف فواصل طبيعية [pause: 2s] عند الحاجة
    - ابدأ كل جملة بـ المقدم: أو الضيف:
    - تأكد من الانتقال الطبيعي من السياق السابق
    """
    response = llm.invoke(prompt)
    state['outro'] = response.content
    return state


workflow = StateGraph(PodState)

workflow.add_node("classify_topic", classify_topic)
workflow.add_node("persona_gen", persona_gen)
workflow.add_node("base_outline_gen", base_outline_gen)
workflow.add_node("outline_enhance_style", outline_enhance_style)
workflow.add_node("outline_enhance_spontanity", outline_enhance_spontanity)
workflow.add_node("intro_gen", intro_gen)
workflow.add_node("script_gen", script_gen)
workflow.add_node("outro_gen", outro_gen)

workflow.add_edge(START, "classify_topic")
workflow.add_edge("classify_topic", "persona_gen")
workflow.add_edge("persona_gen", "base_outline_gen")
workflow.add_edge("base_outline_gen", "outline_enhance_style")
workflow.add_edge("outline_enhance_style", "outline_enhance_spontanity")
workflow.add_edge("outline_enhance_spontanity", "intro_gen")
workflow.add_edge("intro_gen", "script_gen")
workflow.add_edge("script_gen", "outro_gen")
workflow.add_edge("outro_gen", END)

podcast_script_graph = workflow.compile()


'''Topic: "Quantum Computing and its Future Implications"
host_persona: {
  "name": "ليلى القاسمي",
  "age": 42,
  "personality_summary": "مقدمة برامج حيوية وجذابة، بارعة في إدارة الحوار وتحفيز الضيوف، تتمتع بحس فكاهي وقدرة على التواصل مع جمهور متنوع.",
  "OCEAN": {
    "Openness": "High",
    "Conscientiousness": "Moderate",
    "Extraversion": "High",
    "Agreeableness": "High",
    "Neuroticism": "Low"
  },
  "background": "مقدمة برنامج حواري رائد على قناة تلفزيونية إقليمية، خبرة تفوق 15 عامًا في الإعلام، شاركت في تنظيم فعاليات ومهرجانات ثقافية وحفلات حوارية مع قادة الرأي والمشاهير."
}

guest_persona: {
  "name": "د. أمير السعيد",
  "age": 38,
  "personality_summary": "خبير في الحوسبة الكمومية دقيق التحليل، مستكشف للتقنيات المستقبلية ويحب تبسيط المفاهيم المعقدة للجمهور.",
  "OCEAN": {
    "Openness": "High",
    "Conscientiousness": "High",
    "Extraversion": "Moderate",
    "Agreeableness": "Moderate",
    "Neuroticism": "Low"
  },
  "background": "أستاذ مشارك في معهد تكنولوجيا في أبوظبي، مؤسس مختبر أبحاث الحوسبة الكمومية، ومتحدث رئيسي في مؤتمرات دولية حول التشفير الكمومي ومحاكاة الأنظمة الفيزيائية."
}'''



def generate_podcast_script(topic: str) -> str:
    Init_State: PodState = {
        "messages": [],
        "topic": topic,
        "host_persona":  """فيصل العتيبي - العمر: 38
المهنة/الخلفية: إعلامي ومقدم برامج اجتماعية من الرياض، متزوج وأب لثلاثة أطفال
الشخصية: دبلوماسي، مرح، يجيد التعامل مع المواضيع الحساسة بخفة دم، يحب سرد القصص الشخصية
OCEAN: انفتاح عالٍ، ضمير عالٍ، انبساط عالٍ جداً، وداعة عالية، عصابية منخفضة
أسلوب التحدث: يستخدم الفكاهة لكسر الحرج، يشارك تجاربه الشخصية، يطرح أسئلة مباشرة لكن بلطف"""
,
        "guest_persona": """د. نورا السالم - العمر: 42
المهنة/الخلفية: استشارية علم اجتماع وخبيرة في قضايا المرأة والأسرة من بيروت، عزباء بالاختيار
الشخصية: ذكية، صريحة، واثقة من نفسها، تملك حس دعابة عالي، لا تخجل من مناقشة المواضيع الشائكة
OCEAN: انفتاح عالٍ جداً، ضمير عالٍ، انبساط متوسط، وداعة متوسطة، عصابية منخفضة
أسلوب التحدث: تحليل علمي مع لمسة شخصية، تستخدم البيانات والإحصائيات، تتحدث بصراحة عن تجربتها"""
,
        "outline": "",
        "style": "ترفيهي",
        "clutural": True,
        "information": background,
        "factual": False,
        "complexity": "متوسط",
    }
    result = podcast_script_graph.invoke(Init_State)
    print(result)

    return result


def write_script_to_file(script, filename: str = "podcast_script.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        script = str(script)
        f.write(script)
write_script_to_file(generate_podcast_script("ظاهرة العنوسة في المجتمع العربي: أسباب وحلول")['outline'])


