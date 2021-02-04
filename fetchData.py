# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import json

import firebase_admin
from firebase import firebase
from firebase_admin import credentials, firestore

firebase = firebase.FirebaseApplication('https://iotprojectitfinalyear-default-rtdb.europe-west1.firebasedatabase.app',
                                        None)

# Fetch the service account key JSON file contents
cred = credentials.Certificate('./iotprojectitfinalyear-firebase-adminsdk-8mvdp-65d04fe9a4.json')

# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred)

db = firestore.client()

# As an admin, the app has access to read and write all data, regradless of Security Rulesdb = firestore.client()


with open("out.csv", "w", encoding='utf-8') as f:
    f.write("time, Fahrenheit,humidity,moisture,temperature, solenoid_valve\n")


def loadData(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    results = firebase.get('/FirebaseIOT/2021/', '')
    r = remove_empty_from_dict(results)
    _data = json.dumps(r, indent=2)
    jsonDict = json.loads(_data)
    print(f"The result _data {_data}")
    # rf = r["February"]["2021-2-3"]["2"]
    time = ""
    d = ""
    h = ""
    t = ""
    doc_ref = db.collection(u'iotData')

    for feb in r:
        for date in r[feb]:
            print(f" day element {date}")
            time = ""
            time = str(date)

            for day in r[feb][date]:
                d = str(day).replace("_hours", "")
                print(f" day element {d}")
                for hour in r[feb][date][day]:
                    h = ""
                    h = str(hour).replace("_minute", "")
                    t = time + "-" + d + ":" + h
                    dt = ""
                    dt = r[feb][date][day][hour]
                    dt["time"] = t

                    print(f" hour element {h} {dt}")
                    with open("out.csv", "a", encoding='utf-8') as f:
                        f.write("%s,%s,%s,%s,%s,%s\n" % (t,
                                                         r[feb][date][day][hour]["Fahrenheit"],
                                                         r[feb][date][day][hour]["humidity"],
                                                         r[feb][date][day][hour]["moisture"],
                                                         r[feb][date][day][hour]["temperature"],
                                                         r[feb][date][day][hour]["solenoid_valve"]))
                        doc_ref.document(u'' + t).set(dt)
                        t = ""

    # with open("final_year_project.json", "w", encoding='utf-8') as f:
    #     f.write(_data)


def storeCleanedData(cleanData):
    jsonDict = json.loads(cleanData)
    with open("out.csv", "w", encoding='utf-8') as f:
        f.write("Fahrenheit,humidity,moisture,temperature\n")

    print("Clean Data ", jsonDict["0"])
    for item in jsonDict["0"]:
        print(f"this is the item {item}")

        with open("out.csv", "a", encoding='utf-8') as f:
            f.write("%s,%s,%s,%s, %s\n" % (
                item["Fahrenheit"], item["humidity"], item["moisture"],
                item["temperature"]))


def remove_empty_from_dict(d):
    if type(d) is dict:
        return dict((k, remove_empty_from_dict(v)) for k, v in d.items() if v and remove_empty_from_dict(v))
    elif type(d) is list:
        return [remove_empty_from_dict(v) for v in d if v and remove_empty_from_dict(v)]
    else:
        return d


loadData("fetch data")
