import pandas
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = pandas.read_csv("spam.csv", encoding="latin1")
data = data[['v1', 'v2']].rename(columns={"v1":"isSpam", "v2":"content"}).drop_duplicates()
data["isSpam"] = data['isSpam'].replace(["ham", "spam"], [0, 1])

Xdata = data["content"]
Ydata = data["isSpam"]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata, test_size=0.2, random_state=123, stratify=Ydata)
print(f'정상 메일 = {round(Ytest.value_counts()[0]/len(Ytest) * 100,3)}%')
print(f'스팸 메일 = {round(Ytest.value_counts()[1]/len(Ytest) * 100,3)}%')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
