import pandas as pd
from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import requests
import numpy as np
import pickle as p
import json


app = Flask(__name__)



@app.route("/")
def home():
    return render_template("index.html")



@app.route('/predict', methods=["GET", "POST"])
def predict():
    print(request.form)
    if request.method == "POST":
        print("Inside Post")
        PROGRAM_SEMESTERS = request.form.get("PROGRAM_SEMESTERS")
        TOTAL_PROGRAM_SEMESTERS = request.form.get("TOTAL_PROGRAM_SEMESTERS")
        FIRST_YEAR_PERSISTENCE_COUNT = request.form.get("FIRST_YEAR_PERSISTENCE_COUNT")
        #ENGLISH_TEST_SCORE = request.form.get("ENGLISH_TEST_SCORE")
        Term1 = request.form.get("Term1")
        Term2 = request.form.get("Term2")
        Term3 = request.form.get("Term3")
        Term4 = request.form.get("Term4")
        Term5 = request.form.get("Term5")
        Term6 = request.form.get("Term6")
        Term7 = request.form.get("Term7")
        Term8 = request.form.get("Term8")
        Term9 = request.form.get("Term9")
        Term10 = request.form.get("Term10")
        INTAKE_COLLEGE_EXPERIENCE = request.form.get("INTAKE_COLLEGE_EXPERIENCE")
        SCHOOL_CODE = request.form.get("SCHOOL_CODE")
        STUDENT_LEVEL_NAME = request.form.get("STUDENT_LEVEL_NAME")
        TIME_STATUS_NAME = request.form.get("TIME_STATUS_NAME")
        RESIDENCY_STATUS_NAME = request.form.get("RESIDENCY_STATUS_NAME")
        FUNDING_SOURCE_NAME = request.form.get("FUNDING_SOURCE_NAME")
        GENDER = request.form.get("GENDER")
        DISABILITY_IND = request.form.get("DISABILITY_IND")
        MAILING_COUNTRY_NAME = request.form.get("MAILING_COUNTRY_NAME")
        CURRENT_STAY_STATUS = request.form.get("CURRENT_STAY_STATUS")
        ACADEMIC_PERFORMANCE = request.form.get("ACADEMIC_PERFORMANCE")
        AGE_GROUP_LONG_NAME = request.form.get("AGE_GROUP_LONG_NAME")
        APPLICANT_CATEGORY_NAME = request.form.get("APPLICANT_CATEGORY_NAME")
        APPLICANT_TARGET_SEGMENT_NAME = request.form.get("APPLICANT_TARGET_SEGMENT_NAME")
        PREV_EDU_CRED_LEVEL_NAME = request.form.get("PREV_EDU_CRED_LEVEL_NAME")



        PROGRAM_SEMESTERS_array=numerate("PROGRAM_SEMESTERS",PROGRAM_SEMESTERS)
        TOTAL_PROGRAM_SEMESTERS_array=numerate("TOTAL_PROGRAM_SEMESTERS",TOTAL_PROGRAM_SEMESTERS)
        FIRST_YEAR_PERSISTENCE_COUNT_array=numerate("FIRST_YEAR_PERSISTENCE_COUNT",FIRST_YEAR_PERSISTENCE_COUNT)
        Term1_array=numerate("Term1",Term1)
        Term2_array=numerate("Term1",Term2)
        Term3_array=numerate("Term1",Term3)
        Term4_array=numerate("Term1",Term4)
        Term5_array=numerate("Term1",Term5)
        Term6_array=numerate("Term1",Term6)
        Term7_array=numerate("Term1",Term7)
        Term8_array=numerate("Term1",Term8)
        Term9_array=numerate("Term1",Term9)
        Term10_array=numerate("Term1",Term10)
        INTAKE_COLLEGE_EXPERIENCE_array = onehot_encode("INTAKE_COLLEGE_EXPERIENCE", INTAKE_COLLEGE_EXPERIENCE)
        SCHOOL_CODE_array = onehot_encode("SCHOOL_CODE", SCHOOL_CODE)
        STUDENT_LEVEL_NAME_array = onehot_encode("STUDENT_LEVEL_NAME", STUDENT_LEVEL_NAME)
        TIME_STATUS_NAME_array = onehot_encode("TIME_STATUS_NAME", TIME_STATUS_NAME)
        TIME_STATUS_NAME_array = onehot_encode("RESIDENCY_STATUS_NAME", RESIDENCY_STATUS_NAME)
        FUNDING_SOURCE_NAME_array = onehot_encode("FUNDING_SOURCE_NAME", FUNDING_SOURCE_NAME)
        GENDER_array = onehot_encode("GENDER", GENDER)
        DISABILITY_IND_array = onehot_encode("DISABILITY_IND", DISABILITY_IND)
        MAILING_COUNTRY_NAME_array = onehot_encode("MAILING_COUNTRY_NAME", MAILING_COUNTRY_NAME)
        CURRENT_STAY_STATUS_array = onehot_encode("CURRENT_STAY_STATUS", CURRENT_STAY_STATUS)
        ACADEMIC_PERFORMANCE_array = onehot_encode("ACADEMIC_PERFORMANCE", ACADEMIC_PERFORMANCE)
        AGE_GROUP_LONG_NAME_array = onehot_encode("AGE_GROUP_LONG_NAME", AGE_GROUP_LONG_NAME)
        APPLICANT_CATEGORY_NAME_array = onehot_encode("APPLICANT_CATEGORY_NAME", APPLICANT_CATEGORY_NAME)
        APPLICANT_TARGET_SEGMENT_NAME_array = onehot_encode("APPLICANT_TARGET_SEGMENT_NAME", APPLICANT_TARGET_SEGMENT_NAME)
        PREV_EDU_CRED_LEVEL_NAME_array = onehot_encode("PREV_EDU_CRED_LEVEL_NAME", PREV_EDU_CRED_LEVEL_NAME)

        
       #data_array1 = [list(PROGRAM_SEMESTERS) + list(TOTAL_PROGRAM_SEMESTERS) +list(FIRST_YEAR_PERSISTENCE_COUNT) + list(ENGLISH_TEST_SCORE) + list(Term1)+ list(Term2)+ list(Term3)+ list(Term4)+ list(Term5)+ list(Term6)+ list(Term7)+ list(Term8)+ list(Term9)+ list(Term10)]
        data_array=[PROGRAM_SEMESTERS_array + TOTAL_PROGRAM_SEMESTERS_array +FIRST_YEAR_PERSISTENCE_COUNT_array + Term1_array+ Term2_array+ Term3_array+ Term4_array+ Term5_array+ Term6_array+ Term7_array+ Term8_array+ Term9_array+ Term10_array+ INTAKE_COLLEGE_EXPERIENCE_array + SCHOOL_CODE_array+ STUDENT_LEVEL_NAME_array+ TIME_STATUS_NAME_array+ TIME_STATUS_NAME_array+ FUNDING_SOURCE_NAME_array+ GENDER_array+ DISABILITY_IND_array+ MAILING_COUNTRY_NAME_array+ CURRENT_STAY_STATUS_array+ ACADEMIC_PERFORMANCE_array+ AGE_GROUP_LONG_NAME_array+ APPLICANT_CATEGORY_NAME_array+ APPLICANT_TARGET_SEGMENT_NAME_array+ PREV_EDU_CRED_LEVEL_NAME_array]
        print(data_array)

    

    modelfile = 'models/nn_group1.pickle'
    model = p.load(open(modelfile, 'rb'))
    predict = model.predict(data_array)

    return render_template("prediction.html", pred = predict[0])


def numerate(feature,value):
    if feature=="PROGRAM_SEMESTERS":
        array=[0]
        if(value==1):
            array[0]=1
        elif(value==2):
            array[0]=2
        elif(value==3):
            array[0]=3
        elif(value==4):
            array[0]=4
        elif(value==5):
            array[0]=5
        elif(value==6):
            array[0]=6

    elif feature=="TOTAL_PROGRAM_SEMESTERS":
        array=[0]
        if(value==1):
            array[0]=1
        elif(value==2):
            array[0]=2
        elif(value==3):
            array[0]=3
        elif(value==4):
            array[0]=4
        elif(value==5):
            array[0]=5
        elif(value==6):
            array[0]=6
        elif(value==7):
            array[0]=7
        elif(value==8):
            array[0]=8
        elif(value==9):
            array[0]=9
    elif feature=="FIRST_YEAR_PERSISTENCE_COUNT":
        array=[0]
        if(value==0):
            array[0]=0
        elif(value==1):
            array[0]=1
    elif feature=="Term1":
        array=[0]
        if(value==1):
            array[0]=1
        elif(value==2):
            array[0]=2
    elif feature=="Term2":
        array=[0]
        if(value==1):
            array[0]=1
        elif(value==2):
            array[0]=2
    elif feature=="Term3":
        array=[0]
        if(value==1):
            array[0]=1
        elif(value==2):
            array[0]=2
    elif feature=="Term4":
        array=[0]
        if(value==1):
            array[0]=1
        elif(value==2):
            array[0]=2
    elif feature=="Term5":
        array=[0]
        if(value==1):
            array[0]=1
        elif(value==2):
            array[0]=2
    elif feature=="Term6":
        array=[0]
        if(value==1):
            array[0]=1
        elif(value==2):
            array[0]=2
    elif feature=="Term7":
        array=[0]
        if(value==1):
            array[0]=1
        elif(value==2):
            array[0]=2
    elif feature=="Term8":
        array=[0]
        if(value==1):
            array[0]=1
        elif(value==2):
            array[0]=2
    elif feature=="Term9":
        array=[0]
        if(value==1):
            array[0]=1
        elif(value==2):
            array[0]=2
    elif feature=="Term10":
        array=[0]
        if(value==1):
            array[0]=1
        elif(value==2):
            array[0]=2    

    return array




def onehot_encode(feature, value)->list:

    array=[]
    if feature == "INTAKE_COLLEGE_EXPERIENCE":
        array = [0,0,0,0,0,0]
        options = ['Prep Program Enrolled', 'CE Enrolled', 'New to College' ,'Enrolled','Graduate' ,'Prep Program Graduate']
        for i in range(len(options)):
            if options[i] == value:
                array[i] = 1
    elif feature == "SCHOOL_CODE":
        array = [0,0,0,0,0,0,0]
        options = ['CA', 'CH', 'BU', 'AS', 'ST', 'HT', 'TR']
        for i in range(len(options)):
            if options[i] == value:
                array[i] = 1
    elif feature == "STUDENT_LEVEL_NAME":
        array = [0,0]
        options = ['Post Secondary', 'Post Diploma']
        for i in range(len(options)):
            if options[i] == value:
                array[i] = 1
    elif feature == "TIME_STATUS_NAME":
        array = [0,0]
        options = ['Full-Time', 'Part-Time']
        for i in range(len(options)):
            if options[i] == value:
                array[i] = 1
    elif feature == "RESIDENCY_STATUS_NAME":
        array = [0,0]
        options = ['Resident', 'Non-Resident']
        for i in range(len(options)):
            if options[i] == value:
                array[i] = 1
    elif feature == "FUNDING_SOURCE_NAME":
        array = [0,0,0,0,0]
        options = ['GPOG - FT', 'GPOG - PT', 'Intl - Regular', 'Second Career Program', 'Apprentice - PS']
        for i in range(len(options)):
            if options[i] == value:
                array[i] = 1
    elif feature == "GENDER":
        array = [0,0,0]
        options = ['M', 'F','N']
        for i in range(len(options)):
            if options[i] == value:
                array[i] = 1
    elif feature == "DISABILITY_IND":
        array = [0,0]
        options = ['Y','N']
        for i in range(len(options)):
            if options[i] == value:
                array[i] = 1
    elif feature == "MAILING_COUNTRY_NAME":
        array = [0]
        options = ['Canada']
        for i in range(len(options)):
            if options[i] == value:
                array[i] = 1
    elif feature == "CURRENT_STAY_STATUS":
        array = [0,0,0]
        options = ['Graduated','Left College','Completed']
        for i in range(len(options)):
            if options[i] == value:
                array[i] = 1
    elif feature == "ACADEMIC_PERFORMANCE":
        array = [0,0,0,0]
        options = ['AB - Good', 'DF - Poor', 'C - Satisfactory', 'ZZ - Unknown']
        for i in range(len(options)):
            if options[i] == value:
                array[i] = 1
    elif feature == "AGE_GROUP_LONG_NAME":
        array = [0,0,0,0,0,0,0]
        options = ['21 to 25' ,'36 to 40', '26 to 30', '19 to 20', '0 to 18', '31 to 35','41 to 50']
        for i in range(len(options)):
            if options[i] == value:
                array[i] = 1
    elif feature == "APPLICANT_CATEGORY_NAME":
        array = [0,0,0,0,0]
        options = ['High School, Domestic', 'Mature: Domestic  With Post Secondary',
        'Mature: Domestic 19 or older No Academic History',
        'International Student, with Post Secondary', 'BScN, High School Domestic']
        for i in range(len(options)):
            if options[i] == value:
                array[i] = 1
    elif feature == "APPLICANT_TARGET_SEGMENT_NAME":
        array = [0,0,0,0]
        options = ['Non-Direct Entry' 'Direct Entry' 'International' 'Unknown']
        for i in range(len(options)):
            if options[i] == value:
                array[i] = 1
    elif feature == "PREV_EDU_CRED_LEVEL_NAME":
        array = [0,0,0]
        options = ['High School', 'Post Secondary' ,'Unclassified']
        for i in range(len(options)):
            if options[i] == value:
                array[i] = 1
                
    
    return array

if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)










