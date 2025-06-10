import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

def load_mmlu_data():
    """
    Carrega o dataset MMLU e prepara os dados para classificação
    """
    print("Carregando dataset MMLU...")

    
    dataset = load_dataset("cais/mmlu", "all")

    
    all_data = []
    all_subjects = []

    for split in ['train', 'validation', 'test']:
        if split in dataset:
            for item in dataset[split]:
    
                question = item['question']
                choices = item['choices']
                text = f"{question} {' '.join(choices)}"
    
                all_data.append(text)
                all_subjects.append(item['subject'])

    
    df = pd.DataFrame({
        'text': all_data,
        'subject': all_subjects
    })

    print(f"Dataset carregado com {len(df)} amostras e {df['subject'].nunique()} temas únicos")
    print(f"Temas disponíveis: {sorted(df['subject'].unique())}")

    return df

def prepare_features(df, max_features=5000):
    """
    Prepara as features usando TF-IDF
    """
    print("Preparando features com TF-IDF...")

    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',



    )

    X = vectorizer.fit_transform(df['text'])

    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['subject'])

    print(f"Features preparadas: {X.shape}")
    print(f"Número de classes: {len(label_encoder.classes_)}")

    return X, y, vectorizer, label_encoder

def train_random_forest_with_kfold(X, y, n_splits=5):
    """
    Treina RandomForest com validação cruzada K-Fold
    """
    print(f"\nIniciando treinamento com {n_splits}-Fold Cross Validation...")

    
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,


    )

    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    
    print("Executando validação cruzada...")
    cv_results = cross_validate(
        rf_classifier, X, y, 
        cv=kfold, 
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )

    
    results = {}
    for metric in scoring:
        scores = cv_results[f'test_{metric}']
        results[metric] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }

    return results, rf_classifier, kfold

def display_results(results, label_encoder):
    """
    Exibe os resultados da validação cruzada
    """
    print("\n" + "="*60)
    print("RESULTADOS DA VALIDAÇÃO CRUZADA (5-FOLD)")
    print("="*60)

    for metric, stats in results.items():
        print(f"\n{metric.upper()}:")
        print(f"  Média: {stats['mean']:.4f} (±{stats['std']:.4f})")
        print(f"  Scores por fold: {[f'{score:.4f}' for score in stats['scores']]}")

    print(f"\nNúmero total de classes: {len(label_encoder.classes_)}")
    print(f"Classes: {list(label_encoder.classes_[:10])}{'...' if len(label_encoder.classes_) > 10 else ''}")

def train_final_model(X, y, rf_classifier):
    """
    Treina o modelo final com todos os dados
    """
    print("\nTreinando modelo final com todos os dados...")
    final_model = rf_classifier.fit(X, y)

    
    y_pred = final_model.predict(X)
    accuracy = accuracy_score(y, y_pred)

    print(f"Acurácia no conjunto completo: {accuracy:.4f}")

    return final_model

def get_feature_importance(model, vectorizer, top_n=20):
    """
    Analisa a importância das features
    """
    print(f"\nTOP {top_n} FEATURES MAIS IMPORTANTES:")
    print("-" * 50)

    
    feature_names = vectorizer.get_feature_names_out()

    
    importances = model.feature_importances_

    
    indices = np.argsort(importances)[::-1]

    for i in range(min(top_n, len(feature_names))):
        idx = indices[i]
        print(f"{i+1:2d}. {feature_names[idx]:20s} ({importances[idx]:.6f})")

def main():
    """
    Função principal que executa todo o pipeline
    """
    try:
    
        df = load_mmlu_data()

    
        X, y, vectorizer, label_encoder = prepare_features(df)

    
        cv_results, rf_classifier, kfold = train_random_forest_with_kfold(X, y, n_splits=5)

    
        display_results(cv_results, label_encoder)

    
        final_model = train_final_model(X, y, rf_classifier)

    
        get_feature_importance(final_model, vectorizer)
    
        print("\n" + "="*60)
        print("PIPELINE CONCLUÍDO COM SUCESSO!")
        print("="*60)
    
        return final_model, vectorizer, label_encoder, cv_results

    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")
        return None, None, None, None


def predict_subject(model, vectorizer, label_encoder, text):
    """
    Faz predição do tema para um texto novo
    """

    text_vectorized = vectorizer.transform([text])

    
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]

    
    predicted_subject = label_encoder.inverse_transform([prediction])[0]
    confidence = np.max(probability)

    return predicted_subject, confidence

if __name__ == "__main__":

    model, vectorizer, label_encoder, results = main()

    
    if model is not None:
        print("\n" + "="*60)
        print("EXEMPLO DE PREDIÇÃO")
        print("="*60)
    
        example_text = "What is the derivative of x^2 with respect to x?"
        p, confidence = predict_subject(
            model, vectorizer, label_encoder, example_text
        )



        