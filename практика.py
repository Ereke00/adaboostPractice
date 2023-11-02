import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Загрузите данные из CSV файла (замените 'полный_путь_к_вашему_файлу.csv' на фактический путь к вашему файлу данных)
data = pd.read_csv("C:/Users/seker/OneDrive/Рабочий стол/уроки/diabetes.csv")

# Предполагается, что ваши данные содержат признаки и целевую переменную "Outcome"
X = data.drop(columns=['Outcome'])  # Признаки
y = data['Outcome']  # Целевая переменная

# Разделите данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создайте базовый классификатор (дерево решений)
base_classifier = DecisionTreeClassifier(max_depth=1)

# Создайте модель AdaBoost с базовым классификатором
adaboost_model = AdaBoostClassifier(base_classifier, n_estimators=50, random_state=42)

# Обучите модель на обучающих данных
adaboost_model.fit(X_train, y_train)

# Оцените точность модели на тестовых данных
y_pred = adaboost_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Точность модели AdaBoost:", accuracy)

# Визуализируйте результаты
plt.figure(figsize=(12, 4))

# График точности
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Матрица ошибок')
plt.xlabel('Предсказанные значения')
plt.ylabel('Фактические значения')

# Матрица ошибок
plt.subplot(1, 2, 2)
accuracy = accuracy_score(y_test, y_pred)
plt.title('Точность: {:.2f}'.format(accuracy))
plt.show()
