import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

# Sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── PATHS ────────────────────────────────....
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

# ══════════════════════════════════════════════════════════════
#  PART 1 — SYNTHETIC DEFENSE DATASET GENERATION (أرقام ومؤشرات)
# ══════════════════════════════════════════════════════════════

def generate_defense_dataset(n_samples=2000):
    """
    Generate synthetic but statistically realistic defense risk dataset.
    Each row represents a country-year observation.
    """
    print("\n[DATA] Generating synthetic defense dataset...")

    data = []

    for _ in range(n_samples):
        risk_class = np.random.choice([0, 1, 2, 3], p=[0.35, 0.30, 0.22, 0.13])

        mil_exp = {
            0: np.random.uniform(0.5, 15),
            1: np.random.uniform(5, 50),
            2: np.random.uniform(20, 120),
            3: np.random.uniform(40, 200),
        }[risk_class] + np.random.normal(0, 2)

        mil_gdp_pct = {
            0: np.random.uniform(0.5, 2.5),
            1: np.random.uniform(1.8, 4.5),
            2: np.random.uniform(3.5, 8.0),
            3: np.random.uniform(5.0, 36.0),
        }[risk_class] + np.random.normal(0, 0.3)

        arms_import = {
            0: np.random.uniform(0.0, 2.5),
            1: np.random.uniform(1.5, 5.0),
            2: np.random.uniform(4.0, 8.0),
            3: np.random.uniform(6.5, 10.0),
        }[risk_class] + np.random.normal(0, 0.4)

        psi = {
            0: np.random.uniform(0.2, 2.5),
            1: np.random.uniform(-0.8, 0.8),
            2: np.random.uniform(-1.8, -0.3),
            3: np.random.uniform(-2.5, -1.2),
        }[risk_class] + np.random.normal(0, 0.15)

        acled_events = {
            0: np.random.randint(0, 30),
            1: np.random.randint(15, 150),
            2: np.random.randint(80, 500),
            3: np.random.randint(300, 1200),
        }[risk_class]

        fatalities = {
            0: np.random.randint(0, 100),
            1: np.random.randint(50, 800),
            2: np.random.randint(400, 5000),
            3: np.random.randint(2000, 40000),
        }[risk_class]

        battle_ratio = {
            0: np.random.uniform(0.0, 0.3),
            1: np.random.uniform(0.2, 0.5),
            2: np.random.uniform(0.4, 0.65),
            3: np.random.uniform(0.5, 0.75),
        }[risk_class]

        vac_ratio = np.random.uniform(0.1, 0.4) * (risk_class + 1) / 4
        gdp = np.random.uniform(0.05, 25)

        budget_yoy = {
            0: np.random.uniform(-5, 5),
            1: np.random.uniform(0, 12),
            2: np.random.uniform(8, 30),
            3: np.random.uniform(15, 60),
        }[risk_class] + np.random.normal(0, 1)

        mobilization = {
            0: np.random.choice([0, 1], p=[0.85, 0.15]),
            1: np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2]),
            2: np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3]),
            3: np.random.choice([2, 3, 4], p=[0.1, 0.4, 0.5]),
        }[risk_class]

        cyber_level = {
            0: np.random.choice([0, 1], p=[0.7, 0.3]),
            1: np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2]),
            2: np.random.choice([1, 2, 3], p=[0.2, 0.4, 0.4]),
            3: np.random.choice([2, 3, 4], p=[0.1, 0.3, 0.6]),
        }[risk_class]

        nuclear = {
            0: np.random.choice([0, 1], p=[0.9, 0.1]),
            1: np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15]),
            2: np.random.choice([1, 2, 3], p=[0.3, 0.45, 0.25]),
            3: np.random.choice([2, 3, 4], p=[0.2, 0.4, 0.4]),
        }[risk_class]

        risk_score = (
            risk_class * 22 +
            min(arms_import * 3, 25) +
            max(0, -psi * 8) +
            min(acled_events / 20, 15) +
            min(budget_yoy * 0.5, 10) +
            mobilization * 2 +
            np.random.normal(0, 4)
        )
        risk_score = float(np.clip(risk_score, 0, 100))

        data.append({
            'mil_exp_usd_bn':    round(max(0, mil_exp), 2),
            'mil_gdp_pct':       round(np.clip(mil_gdp_pct, 0, 60), 2),
            'arms_import_index': round(np.clip(arms_import, 0, 10), 2),
            'psi':               round(np.clip(psi, -2.5, 2.5), 3),
            'acled_events_6mo':  max(0, acled_events),
            'fatalities_12mo':   max(0, fatalities),
            'battle_ratio':      round(np.clip(battle_ratio, 0, 1), 3),
            'vac_ratio':         round(np.clip(vac_ratio, 0, 1), 3),
            'gdp_usd_tn':        round(max(0.01, gdp), 3),
            'budget_yoy_pct':    round(budget_yoy, 2),
            'mobilization':      mobilization,
            'cyber_level':       cyber_level,
            'nuclear_posture':   nuclear,
            'risk_class':        risk_class,
            'risk_score':        round(risk_score, 1),
        })

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(DATA_DIR, 'defense_dataset.csv'), index=False)
    print(f"  ✓ Generated {len(df)} samples")
    return df


# ══════════════════════════════════════════════════════════════
#  PART 1.5 — ARABIC NLP DATASET GENERATION (توليد نصوص عربية)
# ══════════════════════════════════════════════════════════════

def generate_infowar_dataset(n_samples=1500):
    """
    Generate synthetic Arabic text dataset for NLP classification.
    Labels: 0=أخبار رسمية (Legitimate), 1=دعاية (Propaganda), 2=تضليل إعلامي (Disinformation), 3=عمليات نفسية (Psyops)
    """
    print("\n[DATA] Generating Arabic NLP dataset...")

    # قوالب عربية مصنفة حسب فئات التصنيف
    templates = {
        0: [  # أخبار رسمية / شرعية
            "أكد المسؤولون أن التمارين العسكرية بالقرب من {loc} تتم وفقاً لالتزامات {treaty}.",
            "سجل المراقبون الدوليون نحو {n} حادثة هذا الشهر في منطقة {loc}, بما يتوافق مع التقارير السابقة.",
            "أصدرت وزارة الدفاع البيانات المالية الفصلية التي تظهر زيادة بنسبة {n}% في مخصصات اللوجستيات.",
            "أكد المتحدث الرسمي تنفيذ عملية تبادل روتينية للقوات بالقرب من حدود {loc}.",
            "تظهر صور الأقمار الصناعية التي تحليلها وجود {n} آلية في قاعدة {loc} العسكرية.",
        ],
        1: [  # دعاية
            "قواتنا الباسلة تصد بنجاح هجمات للعدو وتكبدة خسائر فادحة بقيمة {n} عنصراً بالقرب من {loc}.",
            "تكبد الطرف الآخر خسائر كارثية بلغت {n} فرداً في جبهة {loc} والنصر أصبح قاب قوسين أو أدنى.",
            "تم كشف التدخلات الخارجية في {loc} واعتقال {n} من العناصر بواسطة الأجهزة المختصة.",
            "المجتمع الدولي يعترف بحقنا المشروع في الدفاع عن {loc} ضد أي عدوان محتمل.",
            "معنويات الطرف المقابل تنهار بينما تقدم قواتنا على كافة المحاور في محيط {loc}.",
        ],
        2: [  # تضليل إعلامي
            "عاجل: مقتل وإصابة نحو {n} مدنياً في {loc} والجهات الرسمية تحاول اخفاء الأدلة عن وسائل الإعلام.",
            "وثائق مسربة تثبت أن التعزيزات العسكرية في {loc} هي غطاء لبرنامج محظور.",
            "فيديو متداول يظهر استخدام مواد ممنوعة بالقرب من {loc} وسط عجز الجهات عن التأكيد.",
            "مصادر مجهولة تتحدث عن انشقاق جماعي لـ {n} جندياً من حامية {loc}.",
            "تنفي الجهات الرسمية وقوع خسائر في {loc} بينما تتطابق الصور المنتشرة على منصات التواصل مع روايات أخرى.",
        ],
        3: [  # عمليات نفسية
            "إلى الأفراد المتواجدين في {loc}: سلامتكم مضمونة فقط في حال تسليم أنفسكم فوراً، وقد فعلها {n} قبلك.",
            "أهالي {loc}: لقد تخلت عنكم القيادة، انضموا إلينا لضمان الحماية الكاملة.",
            "العسكريون في قاعدة {loc}: لقد فرت القيادة ومقاومتكم بلا جدوى، وقد تم التأكد من {n} حالة.",
            "النظام في {loc} سيسقط خلال {n} أيام، ومن يستميت سيواجه عواقب وخيمة.",
            "زملائكم في نقطة {loc} قاموا بتسليم مواقعهم، كن أنت رقم {n} في اتخاذ القرار الصحيح.",
        ],
    }

    locations = ['صنعاء', 'الحديدة', 'تعز', 'مارب', 'الجوف', 'صعدة', 'البيضاء', 'شبوة', 'حجة', 'الضالع', 'أبين', 'لاذق']
    treaties = ['الاتفاقية المشتركة', 'المواثيق الدولية', 'البرتوكول الدفاعي', 'التفاهمات الأمنية', 'المعاهدة']

    records = []
    for _ in range(n_samples):
        label = np.random.choice([0, 1, 2, 3], p=[0.30, 0.25, 0.28, 0.17])
        tmpl = np.random.choice(templates[label])
        text = tmpl.format(
            loc=np.random.choice(locations),
            n=np.random.randint(10, 5000),
            treaty=np.random.choice(treaties)
        )
        if np.random.random() < 0.3:
            text += " المتعذر التحقق من هذه المصادر بشكل مستقل."
        records.append({'text': text, 'label': label})

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(DATA_DIR, 'infowar_dataset.csv'), index=False, encoding='utf-8-sig')
    print(f"  ✓ Generated {len(df)} Arabic NLP samples")
    return df


# ══════════════════════════════════════════════════════════════
#  PART 2 — TRAINING PIPELINES (بقية كود النماذج والتدريب)
# ══════════════════════════════════════════════════════════════

FEATURE_COLS = [
    'mil_exp_usd_bn', 'mil_gdp_pct', 'arms_import_index', 'psi',
    'acled_events_6mo', 'fatalities_12mo', 'battle_ratio', 'vac_ratio',
    'gdp_usd_tn', 'budget_yoy_pct', 'mobilization', 'cyber_level', 'nuclear_posture'
]
TARGET_CLASS = 'risk_class'
TARGET_SCORE = 'risk_score'

CLASS_NAMES = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']


def evaluate_classifier(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  Model: {name}")
    print(f"  Accuracy:  {acc*100:.1f}%")
    print(f"  Macro F1:  {report['macro avg']['f1-score']*100:.1f}%")

    return {
        'accuracy':        round(acc * 100, 2),
        'macro_f1':        round(report['macro avg']['f1-score'] * 100, 2),
        'confusion_matrix': cm.tolist(),
    }


def train_risk_models(df):
    print("\n" + "="*60)
    print("  TRAINING RISK CLASSIFICATION MODELS")
    print("="*60)

    X = df[FEATURE_COLS].values
    y = df[TARGET_CLASS].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    results = {}
    trained = {}

    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
    lr.fit(X_train_sc, y_train)
    results['logistic_regression'] = evaluate_classifier('Logistic Regression', lr, X_test_sc, y_test)
    trained['logistic_regression'] = {'model': lr, 'scaler': scaler, 'needs_scaling': True}

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=4,
                                min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results['random_forest'] = evaluate_classifier('Random Forest', rf, X_test, y_test)
    trained['random_forest'] = {'model': rf, 'scaler': scaler, 'needs_scaling': False}

    # 3. XGBoost
    xgb = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, max_features=0.8, random_state=42
    )
    xgb.fit(X_train, y_train)
    results['xgboost'] = evaluate_classifier('XGBoost', xgb, X_test, y_test)
    trained['xgboost'] = {'model': xgb, 'scaler': scaler, 'needs_scaling': False}

    # 4. SVM
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train_sc, y_train)
    results['svm'] = evaluate_classifier('SVM (RBF)', svm, X_test_sc, y_test)
    trained['svm'] = {'model': svm, 'scaler': scaler, 'needs_scaling': True}

    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    with open(os.path.join(MODELS_DIR, 'feature_cols.pkl'), 'wb') as f:
        pickle.dump(FEATURE_COLS, f)

    for name, obj in trained.items():
        path = os.path.join(MODELS_DIR, f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    return results, trained


def train_nlp_model(df_nlp):
    print("\n" + "="*60)
    print("  TRAINING ARABIC NLP CLASSIFIER")
    print("="*60)

    X_text = df_nlp['text'].values
    y_nlp = df_nlp['label'].values
    nlp_labels = ['أخبار رسمية', 'دعاية', 'تضليل إعلامي', 'عمليات نفسية']

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y_nlp, test_size=0.2, random_state=42, stratify=y_nlp
    )

    nlp_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2), max_features=5000,
            sublinear_tf=True, min_df=2, strip_accents='unicode',
        )),
        ('clf', LinearSVC(C=1.0, max_iter=2000, random_state=42))
    ])

    nlp_pipeline.fit(X_train, y_train)
    y_pred = nlp_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    nlp_results = {
        'accuracy': round(acc * 100, 2),
    }

    path = os.path.join(MODELS_DIR, 'nlp_infowar.pkl')
    with open(path, 'wb') as f:
        pickle.dump({'pipeline': nlp_pipeline, 'labels': nlp_labels}, f)
    print("  ✓ Saved Arabic nlp_infowar.pkl")

    return nlp_results


def save_training_report(risk_results, nlp_results):
    report = {
        'generated_at': datetime.now().isoformat(),
        'dataset': {
            'samples': 2000,
            'features': FEATURE_COLS,
            'target_classes': CLASS_NAMES,
        },
        'models': {
            **{k: {kk: vv for kk, vv in v.items() if kk != 'report'}
               for k, v in risk_results.items()},
            'nlp_infowar': nlp_results,
        }
    }

    path = os.path.join(MODELS_DIR, 'training_report.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\n  ✓ Training report saved.")
    return report


if __name__ == '__main__':
    print("\n--- ML Training Pipeline (Arabic NLP Support) ---")
    df_defense = generate_defense_dataset(n_samples=2000)
    df_nlp = generate_infowar_dataset(n_samples=1500)
    risk_results, trained_models = train_risk_models(df_defense)
    nlp_results = train_nlp_model(df_nlp)
    save_training_report(risk_results, nlp_results)
    print("\nTraining Complete Successfully with Arabic Data!")