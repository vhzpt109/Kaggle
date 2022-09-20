from sklearn.preprocessing import MinMaxScaler

def DataProcess(train, test):
    MSSubClass_mapping = {20: 1, 30: 2, 40: 3,
                        45: 4, 50: 5, 60: 6, 70: 7, 75: 8,
                        80: 9, 85: 10, 90: 11, 120: 12, 150: 13,
                        160: 14, 180: 15, 190: 16}
    train['MSSubClass'] = train['MSSubClass'].map(MSSubClass_mapping)
    test['MSSubClass'] = test['MSSubClass'].map(MSSubClass_mapping)

    MSZoning_mapping = {"A": 1, "C": 2, "FV": 3,
                     "I": 4, "RH": 5, "RL": 6, "RP": 7, "RM": 8}
    train['MSZoning'] = train['MSZoning'].map(MSZoning_mapping)
    test['MSZoning'] = test['MSZoning'].map(MSZoning_mapping)

    return train, test