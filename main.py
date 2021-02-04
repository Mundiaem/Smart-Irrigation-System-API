# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import json

import pandas as pd
from IPython.display import display
from firebase import firebase
from pandas import json_normalize  # package for flattening json in pandas df
from sklearn.model_selection import train_test_split
from firebase_admin import credentials, firestore
from firebase_admin import db
import firebase_admin


firebase = firebase.FirebaseApplication('https://iotprojectitfinalyear-default-rtdb.europe-west1.firebasedatabase.app',
                                        None)

# Fetch the service account key JSON file contents
cred = credentials.Certificate('./iotprojectitfinalyear-firebase-adminsdk-8mvdp-65d04fe9a4.json')

# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://iotprojectitfinalyear-default-rtdb.europe-west1.firebasedatabase.app'
})

# As an admin, the app has access to read and write all data, regradless of Security Rules
ref = db.reference('/iotprojectitfinalyear-default-rtdb/')
db = firestore.client()
_data_ = json.dumps(ref.get(), indent=2)
print(f"Ref {ref.get()} and db {db}")
with open("final_year_project_admin_data.json", "a", encoding='utf-8') as f:
    f.write(_data_)

def loadData(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    results = firebase.get('/FirebaseIOT/2021/', '')
    _data = json.dumps(results, indent=2)
    jsonDict = json.loads(_data)
    print(f"The result _data {_data}")
    with open("final_year_project.json", "a", encoding='utf-8') as f:
        f.flush()
        f.write(_data)
    de_nullified_list_json = remove_nulls(jsonDict)
    nycphil = json_normalize(results['2021-2-1'])
    nycphil.head()
    df = pd.DataFrame(nycphil, index=nycphil.keys())
    display(df)
    # print(df.to_markdown())

    json_dump = json.dumps(de_nullified_list_json, indent=2)
    storeCleanedData(json_dump)

    # print(json_dump)
    pandas_data = pd.read_json(json_dump)
    # Creating dummy variable for target i.e label
    label = pd.get_dummies(pandas_data.label).iloc[:, 1:]
    pandas_data = pd.concat([pandas_data, label], axis=1)
    print('The data present in one row of the dataset is')
    print(pandas_data.head(1))
    train = pandas_data.iloc[:, 0:4].values
    test = pandas_data.iloc[:, 4:].values
    # x = np.array(pandas_data.drop(['y'], 1))
    # y = np.array(pandas_data['y'])
    # x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
    # cls = neighbors.KNeighborsClassifier()
    # cls.fit(x_train, y_train)
    # accuracy = cls.score(x_test, y_test)
    # print(accuracy)
    print(f"The pandas head {pandas_data.head()}")
    # df = pd.DataFrame(de_nullified_list_json, index=["q","e","r","o"])
    # print(df.to_markdown())
    # displaying the DataFrame
    # display(df)

    train = pandas_data.iloc[:, 0:4].values
    test = pandas_data.iloc[:, 4:].values

    # Dividing the data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.3)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Importing Decision Tree classifier
    from sklearn.tree import DecisionTreeRegressor
    clf = DecisionTreeRegressor()

    # Fitting the classifier into training set
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    from sklearn.metrics import accuracy_score
    # Finding the accuracy of the model
    a = accuracy_score(y_test, pred)
    print("The accuracy of this model is: ", a * 100)


def analyzeData():
    # Reading the csv file
    data = pd.read_csv('out.csv')
    data.replace('?', -99999, inplace=True)
    data.columns = ['Fahrenheit', 'humidity', 'moisture', 'temperature']
    print(data.head())  # Press the green button in the gutter to run the script.


output = []


def storeCleanedData(cleanData):
    jsonDict = json.loads(cleanData)
    with open("out.csv", "w", encoding='utf-8') as f:
        f.write("Fahrenheit,humidity,moisture,temperature\n")

    print("Clean Data ", jsonDict["0"])
    for item in jsonDict["0"]:
        print(f"this is the item {item}")

        with open("out.csv", "a", encoding='utf-8') as f:
            f.write("%s,%s,%s,%s\n" % (
                        item["Fahrenheit"], item["humidity"], item["moisture"],
                        item["temperature"]))



def remove_nulls(x):
    ignored_values = ['null', '', None]
    x["February"]['2021-2-3'] = filter(lambda x: x['value'] not in ignored_values, x["February"]['2021-2-3'])

    # df = pd.DataFrame(list(d.values()), index=d.keys())
    # print(x)
    to_x = list(x.values()).pop(1)

    return to_x


if __name__ == '__main__':
    with open("out.csv", "w", encoding='utf-8') as f:
        f.write("Fahrenheit,humidity,moisture,temperature\n")
    loadData('IoT Project final year ')
    analyzeData()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
