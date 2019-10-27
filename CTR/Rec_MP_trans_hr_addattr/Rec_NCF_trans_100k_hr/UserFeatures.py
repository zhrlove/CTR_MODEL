import pandas as pd
import numpy as np

path = "ml-100k-userInfo.dat"
def getUserFeature():
    header = ['user_id', 'age', 'gender', 'job']
    df = pd.read_csv(path, sep=',', names=header, engine='python')
    user_gender = df.gender
    user_age = df.age
    user_job = df.job
    user_id = df.user_id

    length = len(df)

    user_vectors = {}
    for i in range(length):
        # gender
        gender_vector = [0] * 2
        if user_gender[i] == 'M':
            gender_vector[0] = 1
        else:
            gender_vector[1] = 1

        # age
        age_vector = [0] * 7
        age = int(user_age[i])
        if age == 1:
            age_vector[0] = 1
        elif age  == 18:
            age_vector[1] = 1
        elif age  == 25:
            age_vector[2] = 1
        elif age  == 35:
            age_vector[3] = 1
        elif age  == 45:
            age_vector[4] = 1
        elif age  == 50:
            age_vector[5] = 1
        else:
            age_vector[6] = 1

        # job
        job_vector = [0] * 21
        job = int(user_job[i])
        if job == 0:
            job_vector[0] = 1
        elif job == 1:
            job_vector[1] = 1
        elif job == 2:
            job_vector[2] = 1
        elif job == 3:
            job_vector[3] = 1
        elif job == 4:
            job_vector[4] = 1
        elif job == 5:
            job_vector[5] = 1
        elif job == 6:
            job_vector[6] = 1
        elif job == 7:
            job_vector[7] = 1
        elif job == 8:
            job_vector[8] = 1
        elif job == 9:
            job_vector[9] = 1
        elif job == 10:
            job_vector[10] = 1
        elif job == 11:
            job_vector[11] = 1
        elif job == 12:
            job_vector[12] = 1
        elif job == 13:
            job_vector[13] = 1
        elif job == 14:
            job_vector[14] = 1
        elif job == 15:
            job_vector[15] = 1
        elif job == 16:
            job_vector[16] = 1
        elif job == 17:
            job_vector[17] = 1
        elif job == 18:
            job_vector[18] = 1
        elif job == 19:
            job_vector[19] = 1
        elif job == 20:
            job_vector[20] = 1

        user_vector = gender_vector + age_vector + job_vector

        user_vectors[int(user_id[i]) - 1] = user_vector
    return user_vectors