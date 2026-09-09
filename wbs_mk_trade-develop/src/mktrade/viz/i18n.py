"""Internationalization (i18n) for English and Macedonian."""

from __future__ import annotations

TRANSLATIONS: dict[str, dict[str, str]] = {
    # ── App chrome ──
    "app_title": {
        "en": "North Macedonia Export Opportunity Explorer",
        "mk": "Истражувач на извозни можности на Северна Македонија",
    },
    "app_subtitle": {
        "en": "Knowledge Graph + GNN link prediction for discovering export-expansion opportunities for North Macedonia.",
        "mk": "Граф на знаење + GNN предвидување за откривање извозни можности за Северна Македонија.",
    },
    "language": {"en": "Language", "mk": "Јазик"},
    "nav_header": {"en": "Navigation", "mk": "Навигација"},

    # ── Page names ──
    "page_overview": {"en": "Overview", "mk": "Преглед"},
    "page_task_a": {"en": "Task A: Products", "mk": "Задача А: Производи"},
    "page_task_b": {"en": "Task B: Markets", "mk": "Задача Б: Пазари"},
    "page_models": {"en": "Model Comparison", "mk": "Споредба на модели"},
    "page_explanations": {"en": "Explanations", "mk": "Објаснувања"},
    "page_glossary": {"en": "Glossary", "mk": "Поимник"},

    # ── Overview page ──
    "focus_country": {"en": "Focus Country", "mk": "Фокусирана земја"},
    "mkd_name": {"en": "North Macedonia (MKD)", "mk": "Северна Македонија (МКД)"},
    "best_model": {"en": "Best Model", "mk": "Најдобар модел"},
    "product_opportunities": {"en": "Product Opportunities", "mk": "Производни можности"},
    "run_phase5_first": {"en": "Run Phase 5 first", "mk": "Прво извршете Фаза 5"},
    "overview_header": {"en": "Project Overview", "mk": "Преглед на проектот"},
    "overview_what": {"en": "What does this dashboard show?", "mk": "Што прикажува оваа табла?"},
    "overview_what_body": {
        "en": """This tool predicts **new export opportunities** for North Macedonia using Graph Neural Networks
trained on international trade data. It answers two questions:

- **Task A (Product Diversification):** Which new *products* could MKD start exporting competitively?
- **Task B (Market Expansion):** For products MKD already exports, which new *countries* could it sell to?

The predictions combine machine learning (GNN link prediction on a knowledge graph) with
established trade economics (economic complexity, product proximity, gravity model).""",
        "mk": """Оваа алатка предвидува **нови извозни можности** за Северна Македонија користејќи графовски невронски мрежи
обучени на меѓународни трговски податоци. Одговара на две прашања:

- **Задача А (Диверзификација на производи):** Кои нови *производи* би можела МКД да почне конкурентно да ги извезува?
- **Задача Б (Експанзија на пазари):** За производите што МКД веќе ги извезува, на кои нови *земји* би можела да им продава?

Предвидувањата комбинираат машинско учење (GNN предвидување на врски во граф на знаење) со
воспоставена трговска економија (економска комплексност, близина на производи, гравитациски модел).""",
    },
    "pipeline_summary": {"en": "Pipeline Summary", "mk": "Резиме на процесот"},
    "pipeline_table": {
        "en": """| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data acquisition & cleaning (Atlas, Comtrade, CEPII, WDI) | Done |
| 2 | Economic complexity metrics (RCA, ECI, PCI, proximity) | Done |
| 3 | Knowledge graph construction (7 edge types, 3 node types) | Done |
| 4 | GNN training & evaluation (GAT, HGT, GraphSAGE, GCN) | Done |
| 5 | Opportunity ranking, ensemble & dashboard | Done |""",
        "mk": """| Фаза | Опис | Статус |
|------|------|--------|
| 1 | Прибирање и чистење на податоци (Atlas, Comtrade, CEPII, WDI) | Завршено |
| 2 | Метрики за економска комплексност (RCA, ECI, PCI, близина) | Завршено |
| 3 | Конструкција на граф на знаење (7 типови врски, 3 типови јазли) | Завршено |
| 4 | Обука и евалуација на GNN (GAT, HGT, GraphSAGE, GCN) | Завршено |
| 5 | Рангирање на можности, ансамбл и табла | Завршено |""",
    },
    "model_perf_summary": {"en": "Model Performance Summary", "mk": "Резиме на перформансите на моделите"},
    "model_perf_caption": {
        "en": "AUC and Average Precision on the held-out test set (2021-2022 new exports).",
        "mk": "AUC и просечна прецизност на тест множеството (нови извози 2021-2022).",
    },

    # ── Task A ──
    "task_a_header": {"en": "Task A: Product Diversification", "mk": "Задача А: Диверзификација на производи"},
    "task_a_desc": {
        "en": "Products that North Macedonia **does not yet export competitively** (RCA < 1) but the learned ranking system predicts it could develop a sustained comparative advantage in. Higher final scores indicate stronger predicted fit.",
        "mk": "Производи кои Северна Македонија **сè уште не ги извезува конкурентно** (RCA < 1), но научениот систем за рангирање предвидува дека може да развие одржлива компаративна предност. Повисоките конечни резултати значат посилна предвидена соодветност.",
    },
    "num_products": {"en": "Number of products to show", "mk": "Број на производи за прикажување"},
    "filter_chapter": {"en": "Filter by HS chapter", "mk": "Филтрирај по HS поглавје"},
    "opportunity_details": {"en": "Opportunity Details", "mk": "Детали за можности"},
    "col_product": {"en": "Product", "mk": "Производ"},
    "col_final_score": {"en": "Predicted Success", "mk": "Предвиден успех"},
    "col_gnn_score": {"en": "GNN Score", "mk": "GNN резултат"},
    "col_density": {"en": "Density", "mk": "Густина"},
    "col_pci": {"en": "PCI", "mk": "PCI"},
    "col_chapter": {"en": "Chapter", "mk": "Поглавје"},
    "what_columns_mean": {"en": "What do these columns mean?", "mk": "Што значат овие колони?"},
    "task_a_columns_help": {
        "en": """| Column | Description |
|--------|-------------|
| **Product** | HS4 code and name — the specific product category |
| **Predicted Success** | Validation-calibrated probability from the learned reranker (0-1); this is the final recommendation score |
| **GNN Score** | The GNN's sigmoid link signal (0-1). It is an input to the reranker, not a calibrated success probability |
| **Density** | How close MKD's current exports are to this product in the product space (0-1). Higher = easier transition |
| **PCI** | Product Complexity Index — how sophisticated this product is. Higher = more complex, more valuable |
| **Chapter** | The HS chapter (first 2 digits), grouping related products |""",
        "mk": """| Колона | Опис |
|--------|------|
| **Производ** | HS4 код и име — специфична категорија на производ |
| **Предвиден успех** | Веројатност калибрирана на валидациски податоци од научениот прерангирач (0-1); конечниот резултат за препораките |
| **GNN резултат** | Сигмоиден GNN сигнал за врска (0-1). Тој е влез во прерангирачот, а не калибрирана веројатност за успех |
| **Густина** | Колку се блиски тековните извози на МКД до овој производ во просторот на производи (0-1). Повисока = полесна транзиција |
| **PCI** | Индекс на комплексност на производ — колку е софистициран. Повисок = покомплексен, повреден |
| **Поглавје** | HS поглавје (првите 2 цифри), групирање на слични производи |""",
    },
    "ensemble_header": {"en": "Ensemble Rankings", "mk": "Ансамбл рангирање"},
    "ensemble_caption": {
        "en": "Learned combination of GNN, density, complexity, demand, persistence and classical signals. The final score is calibrated on validation outcomes; ranking is driven by the pairwise reranker.",
        "mk": "Научена комбинација од GNN, густина, комплексност, побарувачка, постојаност и класични сигнали. Конечниот резултат е калибриран на валидациски исходи, а редоследот го одредува парниот прерангирач.",
    },

    # ── Task B ──
    "task_b_header": {"en": "Task B: Market Expansion", "mk": "Задача Б: Експанзија на пазари"},
    "task_b_desc": {
        "en": "For products MKD already exports, these are **new destination countries** the model predicts as promising markets. The score reflects how likely a trade link is based on country similarity, geographic proximity, GDP, and existing trade patterns.",
        "mk": "За производите кои МКД веќе ги извезува, ова се **нови дестинации** кои моделот ги предвидува како ветувачки пазари. Резултатот ја одразува веројатноста за трговска врска базирана на сличност на земји, географска близина, БДП и постоечки трговски обрасци.",
    },
    "num_entries": {"en": "Number of entries to show", "mk": "Број на записи за прикажување"},
    "group_by": {"en": "Group by:", "mk": "Групирај по:"},
    "group_product": {"en": "Product", "mk": "Производ"},
    "group_market": {"en": "Market", "mk": "Пазар"},
    "select_product": {"en": "Select product:", "mk": "Изберете производ:"},
    "select_market": {"en": "Select market:", "mk": "Изберете пазар:"},
    "top_markets_for": {"en": "Top markets for", "mk": "Најдобри пазари за"},
    "top_products_for": {"en": "Top products for market", "mk": "Најдобри производи за пазар"},
    "col_market": {"en": "Market", "mk": "Пазар"},
    "show_full_results": {"en": "Show full Task B results", "mk": "Прикажи ги сите резултати од Задача Б"},
    "task_b_columns_help": {
        "en": """| Column | Description |
|--------|-------------|
| **Product** | HS4 product code and name |
| **Market** | ISO3 country code and country name |
| **GNN Score** | Predicted probability of a trade link forming (0-1) |""",
        "mk": """| Колона | Опис |
|--------|------|
| **Производ** | HS4 код и име на производот |
| **Пазар** | ISO3 код и име на земјата |
| **GNN резултат** | Предвидена веројатност за формирање трговска врска (0-1) |""",
    },

    # ── Model Comparison ──
    "models_header": {"en": "Model Comparison", "mk": "Споредба на модели"},
    "models_desc": {
        "en": "Models learn from historical export transitions, validate on the non-overlapping 2019-2020 outcome window, and test on **new sustained export links in 2021-2022**. The task: rank which country-product pairs will become meaningful, persistent exports.",
        "mk": "Моделите учат од историски извозни премини, се валидираат во непреклопувачкиот период 2019-2020 и се тестираат на **нови одржливи извозни врски во 2021-2022**. Задачата е да се рангира кои парови земја-производ ќе станат значајни и трајни извози.",
    },
    "detailed_metrics": {"en": "Detailed Metrics", "mk": "Детални метрики"},
    "what_metrics_mean": {"en": "What do these metrics mean?", "mk": "Што значат овие метрики?"},
    "metrics_help": {
        "en": """| Metric | Description |
|--------|-------------|
| **ROC AUC** | Area under the ROC curve (0.5 = random, 1.0 = perfect). Measures how well the model separates real exports from non-exports |
| **Avg Precision** | Area under the Precision-Recall curve. Random AP equals the positive rate in the evaluated candidate universe |
| **AP lift** | Average Precision divided by the current test set's positive rate. A value of 1× is random; larger is better |
| **MRR** | Query-based mean reciprocal rank — the average reciprocal rank of the first correct product within each country query |
| **Precision@10** | Of each country's top 10 predictions, how many are actual future successes |
| **MAP@10** | Mean Average Precision in the top 10, rewarding correct recommendations placed earlier |
| **NDCG@10** | Ranking quality in the top 10, giving more credit when correct recommendations appear near the top |
| **Hits@K** | Fraction of true new exports appearing in the top K predictions |
| **Precision@50** | Of the top 50 predictions, how many are actual future exports |
| **Recall@50** | Of all actual future exports, how many appear in the top 50 predictions |""",
        "mk": """| Метрика | Опис |
|---------|------|
| **ROC AUC** | Површина под ROC кривата (0.5 = случајно, 1.0 = совршено). Мери колку добро моделот ги разделува вистинските извози од не-извозите |
| **Avg Precision** | Површина под кривата прецизност-отповикување. Случајниот AP е еднаков на стапката на позитивни примери во оценетиот сет |
| **AP подобрување** | Просечната прецизност поделена со стапката на позитивни примери во тековниот тест. Вредност 1× е случајно рангирање; повисоко е подобро |
| **MRR** | Среден реципрочен ранг по барање — просечниот реципрочен ранг на првиот точен производ за секоја земја |
| **Precision@10** | Од првите 10 предвидувања за секоја земја, колку се вистински идни успеси |
| **MAP@10** | Средна просечна прецизност во првите 10, која ги наградува точните препораки поставени порано |
| **NDCG@10** | Квалитет на рангирањето во првите 10, со поголема тежина за точните препораки при врвот |
| **Hits@K** | Дел од вистинските нови извози кои се појавуваат во врвните K предвидувања |
| **Precision@50** | Од врвните 50 предвидувања, колку се вистински идни извози |
| **Recall@50** | Од сите вистински идни извози, колку се појавуваат во врвните 50 предвидувања |""",
    },

    # ── Explanations ──
    "explain_header": {"en": "Link Explanations", "mk": "Објаснувања на врски"},
    "explain_desc": {
        "en": "**Why** does the model predict a particular export opportunity? Feature importance is computed using gradient-based attribution: features that change the prediction score the most are deemed most important.",
        "mk": "**Зошто** моделот предвидува одредена извозна можност? Важноста на карактеристиките е пресметана со градиентна атрибуција: карактеристиките кои најмногу го менуваат резултатот се сметаат за најважни.",
    },
    "select_opportunity": {"en": "Select opportunity:", "mk": "Изберете можност:"},
    "country_features": {"en": "Country Features (MKD)", "mk": "Карактеристики на земјата (МКД)"},
    "country_features_caption": {
        "en": "How much each MKD characteristic contributes to predicting this export link.",
        "mk": "Колку секоја карактеристика на МКД придонесува за предвидување на оваа извозна врска.",
    },
    "product_features": {"en": "Product Features", "mk": "Карактеристики на производот"},
    "product_features_caption": {
        "en": "How much each product characteristic contributes to the prediction.",
        "mk": "Колку секоја карактеристика на производот придонесува за предвидувањето.",
    },
    "country_features_help_title": {
        "en": "What do these country features mean?",
        "mk": "Што значат карактеристиките на земјата?",
    },
    "country_features_help": {
        "en": """| Feature | Description |
|---------|-------------|
| **eci** | Economic Complexity Index — how diversified and complex MKD's export basket is |
| **diversity** | Number of products MKD has comparative advantage in (RCA > 1) |
| **gdp_log** | Log of GDP (economic size) |
| **gdp_pc_log** | Log of GDP per capita (development level) |
| **pop_log** | Log of population |
| **landlocked** | Whether the country is landlocked (1 = yes) |
| **eu_member** | EU membership (0 for MKD) |
| **cefta_member** | CEFTA free trade area membership (1 for MKD) |
| **region_*** | One-hot encoded world region |""",
        "mk": """| Карактеристика | Опис |
|----------------|------|
| **eci** | Индекс на економска комплексност — колку е разновиден и комплексен извозот на МКД |
| **diversity** | Број на производи со компаративна предност (RCA > 1) |
| **gdp_log** | Логаритам од БДП (економска големина) |
| **gdp_pc_log** | Логаритам од БДП по глава на жител (ниво на развој) |
| **pop_log** | Логаритам од популација |
| **landlocked** | Дали земјата е копнена (1 = да) |
| **eu_member** | Членство во ЕУ (0 за МКД) |
| **cefta_member** | Членство во ЦЕФТА (1 за МКД) |
| **region_*** | One-hot кодирана светска регија |""",
    },
    "product_features_help_title": {
        "en": "What do these product features mean?",
        "mk": "Што значат карактеристиките на производот?",
    },
    "product_features_help": {
        "en": """| Feature | Description |
|---------|-------------|
| **pci** | Product Complexity Index — how much know-how is needed to produce it |
| **ubiquity** | How many countries export this product competitively (lower = more exclusive) |
| **section_N** | One-hot encoded HS section (21 broad product categories) |""",
        "mk": """| Карактеристика | Опис |
|----------------|------|
| **pci** | Индекс на комплексност на производ — колку знаење е потребно за негово производство |
| **ubiquity** | Колку земји го извезуваат овој производ конкурентно (пониска = поексклузивен) |
| **section_N** | One-hot кодирана HS секција (21 широка категорија производи) |""",
    },

    # ── Glossary ──
    "glossary_header": {"en": "Glossary & Reference", "mk": "Поимник и референца"},
    "glossary_intro": {
        "en": "This page explains the codes, metrics, and terminology used throughout the dashboard.",
        "mk": "Оваа страница ги објаснува кодовите, метриките и терминологијата користени во таблата.",
    },
    "hs_code_title": {"en": "The Harmonized System (HS) Code", "mk": "Хармонизиран систем (HS) код"},
    "hs_code_body": {
        "en": """The **Harmonized System (HS)** is an international standard for classifying traded goods,
maintained by the World Customs Organization. Every product traded internationally is assigned a numeric code:

- **HS2** (2 digits): **Chapter** — broad product category (e.g., 85 = Electrical Equipment)
- **HS4** (4 digits): **Heading** — more specific product (e.g., 8544 = Insulated Wire & Cable)
- **HS6** (6 digits): Even more specific (not used in this project)

This dashboard uses **HS4 codes** (1,241 product categories). The first two digits indicate the chapter.
For example, HS 8708 (Vehicle Parts) belongs to Chapter 87 (Vehicles).""",
        "mk": """**Хармонизираниот систем (HS)** е меѓународен стандард за класификација на трговски стоки,
одржуван од Светската царинска организација. На секој производ му е доделен нумерички код:

- **HS2** (2 цифри): **Поглавје** — широка категорија на производ (нпр. 85 = Електрична опрема)
- **HS4** (4 цифри): **Наслов** — поспецифичен производ (нпр. 8544 = Изолирана жица и кабел)
- **HS6** (6 цифри): Уште поспецифичен (не се користи во овој проект)

Оваа табла користи **HS4 кодови** (1.241 категорија на производи). Првите две цифри го означуваат поглавјето.
На пример, HS 8708 (Делови за возила) припаѓа на Поглавје 87 (Возила).""",
    },
    "hs_chapters_title": {"en": "HS Chapters", "mk": "HS Поглавја"},
    "show_all_chapters": {"en": "Show all 97 HS chapters", "mk": "Прикажи ги сите 97 HS поглавја"},
    "hs_sections_title": {"en": "HS Sections (21 broad groups)", "mk": "HS Секции (21 широка група)"},
    "show_all_sections": {"en": "Show all 21 HS sections", "mk": "Прикажи ги сите 21 HS секции"},
    "country_codes_title": {"en": "Country Codes (ISO 3166-1 alpha-3)", "mk": "Кодови на земји (ISO 3166-1 алфа-3)"},
    "country_codes_body": {
        "en": "Countries are identified by **3-letter ISO codes** (e.g., MKD = North Macedonia, DEU = Germany, SRB = Serbia). These are the standard codes used in international trade data.",
        "mk": "Земјите се идентификуваат со **3-буквени ISO кодови** (нпр. MKD = Северна Македонија, DEU = Германија, SRB = Србија). Ова се стандардни кодови користени во меѓународни трговски податоци.",
    },
    "trade_metrics_title": {"en": "Key Trade & Complexity Metrics", "mk": "Клучни трговски метрики и метрики за комплексност"},
    "trade_metrics_body": {
        "en": """| Term | What it means |
|------|---------------|
| **RCA** (Revealed Comparative Advantage) | If RCA > 1, the country exports this product more than the world average — it has a "comparative advantage" |
| **ECI** (Economic Complexity Index) | Measures how diversified and sophisticated a country's exports are. Higher ECI = more complex economy |
| **PCI** (Product Complexity Index) | Measures how much knowledge/capability is needed to produce a product. Higher PCI = more complex product |
| **Density** | How "close" a product is to a country's current export basket in the product space. Range 0-1. Higher = more related |
| **Proximity** | How often two products are co-exported by the same countries. High proximity = similar capabilities needed |
| **Ubiquity** | How many countries export a product with RCA > 1. Low ubiquity = fewer countries can make it |
| **Diversity** | How many products a country exports with RCA > 1. High diversity = broad export basket |""",
        "mk": """| Поим | Значење |
|------|---------|
| **RCA** (Откриена компаративна предност) | Ако RCA > 1, земјата го извезува овој производ повеќе од светскиот просек — има „компаративна предност" |
| **ECI** (Индекс на економска комплексност) | Мери колку се разновидни и софистицирани извозите. Повисок ECI = покомплексна економија |
| **PCI** (Индекс на комплексност на производ) | Мери колку знаење/способност е потребно за производство. Повисок PCI = покомплексен производ |
| **Густина** | Колку е „блиску" производот до тековниот извоз на земјата. Опсег 0-1. Повисока = повеќе поврзан |
| **Близина** | Колку често два производа се ко-извезуваат од истите земји. Висока близина = слични способности |
| **Присутност** | Колку земји го извезуваат производот со RCA > 1. Ниска = помалку земји можат да го направат |
| **Разновидност** | Колку производи земјата ги извезува со RCA > 1. Висока = широк извозен кошница |""",
    },
    "ml_terms_title": {"en": "Model & ML Terminology", "mk": "Терминологија за модели и машинско учење"},
    "ml_terms_body": {
        "en": """| Term | What it means |
|------|---------------|
| **GNN** (Graph Neural Network) | A neural network that operates on graph-structured data. Learns from the network of countries, products, and trade relationships |
| **GAT** (Graph Attention Network) | GNN variant that uses attention mechanisms to weight the importance of different neighbors |
| **HGT** (Heterogeneous Graph Transformer) | GNN designed for graphs with multiple node and edge types |
| **Link Prediction** | The ML task: predict which new edges (trade links) will form in the graph |
| **AUC** (Area Under the ROC Curve) | Model quality metric. 0.5 = random guessing, 1.0 = perfect predictions |
| **Average Precision** | Model quality metric that focuses on ranking — are the most confident predictions correct? |
| **Ensemble** | Combining multiple models/scores (GNN + density + classical) for more robust predictions |""",
        "mk": """| Поим | Значење |
|------|---------|
| **GNN** (Графовска невронска мрежа) | Невронска мрежа што работи на графовски структурирани податоци. Учи од мрежата на земји, производи и трговски врски |
| **GAT** (Мрежа со графовско внимание) | GNN варијанта што користи механизми за внимание за пондерирање на различни соседи |
| **HGT** (Хетероген графовски трансформер) | GNN дизајниран за графови со повеќе типови јазли и врски |
| **Link Prediction** (Предвидување на врски) | ML задача: предвиди кои нови врски (трговски врски) ќе се формираат |
| **AUC** (Површина под ROC кривата) | Метрика за квалитет. 0.5 = случајно погодување, 1.0 = совршени предвидувања |
| **Average Precision** (Просечна прецизност) | Метрика фокусирана на рангирање — дали најсигурните предвидувања се точни? |
| **Ансамбл** | Комбинирање повеќе модели/резултати (GNN + густина + класични) за поробусни предвидувања |""",
    },
    "kg_structure_title": {"en": "Knowledge Graph Structure", "mk": "Структура на графот на знаење"},
    "kg_structure_body": {
        "en": """The knowledge graph has **3 node types** and **7 edge types**:

**Nodes:**
- **Country** (230): Each country with features like ECI, GDP, population, region
- **Product** (1,241): Each HS4 product with features like PCI, ubiquity, section
- **Product Section** (21): Broad HS sections grouping products

**Edges:**
- **exports**: Country → Product (the main link we predict)
- **proximity**: Product ↔ Product (co-export similarity)
- **in_section**: Product → Section (classification hierarchy)
- **trades_with**: Country ↔ Country (bilateral trade partners)
- **neighbor_of**: Country ↔ Country (from gravity model: distance, shared border, language, FTA)
- Plus reverse edges for message passing""",
        "mk": """Графот на знаење има **3 типа јазли** и **7 типа врски**:

**Јазли:**
- **Земја** (230): Секоја земја со карактеристики: ECI, БДП, популација, регија
- **Производ** (1.241): Секој HS4 производ со карактеристики: PCI, присутност, секција
- **Секција** (21): Широки HS секции кои ги групираат производите

**Врски:**
- **exports** (извезува): Земја → Производ (главната врска што ја предвидуваме)
- **proximity** (близина): Производ ↔ Производ (сличност на ко-извоз)
- **in_section** (во секција): Производ → Секција (класификациска хиерархија)
- **trades_with** (тргува со): Земја ↔ Земја (билатерални трговски партнери)
- **neighbor_of** (сосед на): Земја ↔ Земја (од гравитационен модел: далечина, граница, јазик, FTA)
- Плус реверзни врски за пропагација на пораки""",
    },

    # ── Misc ──
    "no_data_warning": {"en": "No results found. Run the appropriate script first.", "mk": "Нема резултати. Прво извршете ја соодветната скрипта."},
    "select_page": {"en": "Select page:", "mk": "Изберете страница:"},
    "target_markets": {"en": "Target Markets", "mk": "Целни пазари"},
    "no_filter_match": {"en": "No products match the selected filters.", "mk": "Нема производи што одговараат на избраните филтри."},
    "sidebar_info": {
        "en": "MSc Thesis Project\n\nKnowledge Graph + GNN\nLink Prediction",
        "mk": "Магистерски проект\n\nГраф на знаење + GNN\nПредвидување на врски",
    },
    "score_label": {"en": "Score", "mk": "Резултат"},
    "tab_heatmap": {"en": "Heatmap", "mk": "Топлотна мапа"},
    "tab_bar_chart": {"en": "Bar Chart", "mk": "Столбест графикон"},
    "tab_ranking_quality": {"en": "Ranking Quality", "mk": "Квалитет на рангирање"},
    "ranking_quality_caption": {
        "en": "AP lift compares each model's Average Precision with the test set's positive rate (random ranking = 1×). The random line applies only to AP lift.",
        "mk": "AP подобрувањето ја споредува просечната прецизност на секој модел со стапката на позитивни примери во тестот (случајно рангирање = 1×). Случајната линија важи само за AP подобрувањето.",
    },

    # ── Ensemble column display names ──
    "col_ensemble_score": {"en": "Ensemble Score", "mk": "Ансамбл резултат"},
    "col_gnn_score_short": {"en": "GNN Score", "mk": "GNN резултат"},
    "col_density_score": {"en": "Density Score", "mk": "Резултат на густина"},
    "col_classical_score": {"en": "Classical Score", "mk": "Класичен резултат"},

    # ── Glossary table headers ──
    "col_chapter_num": {"en": "Chapter", "mk": "Поглавје"},
    "col_name": {"en": "Name", "mk": "Име"},
    "col_section_num": {"en": "Section", "mk": "Секција"},
    "col_description": {"en": "Description", "mk": "Опис"},

    # ── Plot titles and labels ──
    "chart_top_opportunities": {"en": "Top Predicted Product Opportunities for MKD", "mk": "Најдобро предвидени производни можности за МКД"},
    "chart_gnn_link_score": {"en": "GNN Link Score", "mk": "GNN резултат на врска"},
    "chart_opportunity_score": {"en": "Predicted Success Probability", "mk": "Предвидена веројатност за успех"},
    "chart_product": {"en": "Product", "mk": "Производ"},
    "chart_chapter": {"en": "Chapter", "mk": "Поглавје"},
    "chart_hs_chapter": {"en": "HS Chapter", "mk": "HS Поглавје"},
    "chart_model_comparison": {"en": "Model Comparison", "mk": "Споредба на модели"},
    "chart_metric": {"en": "Metric", "mk": "Метрика"},
    "chart_model": {"en": "Model", "mk": "Модел"},
    "chart_model_perf": {"en": "Model Performance Comparison", "mk": "Споредба на перформанси на модели"},
    "chart_auc_comparison": {"en": "ROC-AUC by Model", "mk": "ROC-AUC по модел"},
    "chart_ranking_quality": {"en": "Ranking Quality at the Top of the List", "mk": "Квалитет на рангирање на врвот на листата"},
    "chart_top10_quality": {"en": "Top-10 ranking metrics", "mk": "Метрики за првите 10"},
    "chart_ap_lift": {"en": "Average Precision lift over random", "mk": "Подобрување на просечната прецизност над случајното"},
    "chart_ap_lift_short": {"en": "AP lift", "mk": "AP подобрување"},
    "chart_avg_precision": {"en": "Average Precision", "mk": "Просечна прецизност"},
    "chart_random_baseline": {"en": "Random baseline (1×)", "mk": "Случајна основа (1×)"},
    "chart_lift": {"en": "Lift over random", "mk": "Подобрување над случајното"},
    "chart_score": {"en": "Score", "mk": "Резултат"},
    "chart_no_importance": {"en": "No importance data available", "mk": "Нема достапни податоци за важност"},
    "chart_feature_importance": {"en": "Feature Importance", "mk": "Важност на карактеристики"},
    "chart_importance": {"en": "Importance", "mk": "Важност"},
    "chart_feature": {"en": "Feature", "mk": "Карактеристика"},
    "chart_opportunities_by_chapter": {"en": "Opportunities by HS Chapter", "mk": "Можности по HS поглавје"},
    "chart_country": {"en": "Country", "mk": "Земја"},
    "chart_product_node": {"en": "Product", "mk": "Производ"},
}


def t(key: str, lang: str = "en") -> str:
    """Look up translation, fall back to English, then to the key itself."""
    entry = TRANSLATIONS.get(key, {})
    return entry.get(lang, entry.get("en", key))
