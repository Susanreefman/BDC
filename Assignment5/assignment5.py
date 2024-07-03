#!/usr/bin/env python3

"""
Assignment 5 Big Data Computing BFV3

hsreefman
"""


# Imports
import sys
import io
import csv
from contextlib import redirect_stdout
import pandas
import pyspark.sql.functions as fun
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, desc


# Functions
def create_dataframe(input_file):
    """
    Create a spark dataframe
    param: 
        file (string): input file
    return:
        dataframe (spark.DataFrame): dataframe
    """
    header = ["protein_accession", "seq_MD5_digest", "seq_length", "analysis",
            "signature_accession", "signature_desc", "start_loc", "stop_loc",
            "score", "status", "date", "interpro_annot_accession", "interpro_annot_desc",
            "go_annot", "pathways_annot"]

    spark = SparkSession.builder.appName("Interpro") \
        .master("spark://spark.bin.bioinf.nl:7077").getOrCreate()
    pd_df = pandas.read_csv(input_file, sep='\t', header=0, names=header)
    spark_df = spark.createDataFrame(pd_df)

    return spark_df


# Function to capture explain output as string
def capture_explain(exp):
    """Get the explanation"""
    buf = io.StringIO()
    with redirect_stdout(buf):
        exp.explain(True)
    return buf.getvalue()


def question1(data):
    """
    How many protein annotations are there in this dataset, 
    i.e. how many unique InterPRO numbers are there?
    Param:
        data (spark.DataFrame): dataset
    Return:
        (list) : answer to the question and explain string in a list
    """
    answer = data.select("interpro_annot_accession").distinct().count()
    explain = capture_explain(data.select("interpro_annot_accession").distinct())

    return [answer, explain]


def question2(data):
    """
    How many InterPRO annotations does a protein have on average?
    Param:
        data (spark.DataFrame): dataset
    Return:
        (list) : answer to the question and explain string in a list
    """
    answer = data.groupBy("interpro_annot_accession").count().agg(avg("count")).first()[0]
    explain = capture_explain(data.groupBy("interpro_annot_accession").count().agg(avg("count")))
    return [answer, explain]


def question3(data):
    """
    Of the GO Terms that are also annotated, which is the most common?
    Param:
        data (spark.DataFrame): dataset
    Return:
        (list) : answer to the question and explain string in a list
	"""
    data = data.where(data.go_annot != "-")
    answer = data.groupBy("go_annot").count().orderBy(desc("count")).first()[0]
    explain = capture_explain(data.groupBy("go_annot").count().orderBy(desc("count")))
    print(answer)
    print(explain)

    return [answer, explain]


def question4(data):
    """
    What is the size (in Amino Acids) of an InterPRO feature in this dataset?
    Param:
        data (spark.DataFrame): dataset
    Return:
        (list) : answer to the question and explain string in a list
	"""
    data = data.where(data.interpro_annot_accession != "-")
    answer = data.groupBy("interpro_annot_accession").agg({"seq_length": "mean"}).first()[1]
    explain = capture_explain(data.groupBy("interpro_annot_accession").agg({"seq_length": "mean"}))
    print(answer)
    print(explain)

    return [answer, explain]


def question5(data):
    """
    What are the top 10 most common InterPRO annotations?
    Param:
        data (spark.DataFrame): dataset
    Return:
        (list) : answer to the question and explain string in a list
	"""
    data = data.where(data.interpro_annot_accession != "-")
    answer = data.groupBy("interpro_annot_accession").count() \
        .orderBy(desc("count")).limit(10).collect()
    top_10 = [row.interpro_annot_accession for row in answer]

    explain = capture_explain(data.groupBy("interpro_annot_accession") \
                              .count().orderBy(desc("count")).limit(10))

    return [top_10, explain]


def question6(data):
    """
    If you select for InterPRO annotations that span >90% of the protein, 
    what are the top 10 most frequently found InterPRO annotations?
    Param:
        data (spark.DataFrame): dataset
    Return:
        (list) : answer to the question and explain string in a list
	"""
    data = data.where(data.interpro_annot_accession != "-")
    new_data = data.withColumn("same_size",
                          fun.when((data["stop_loc"] - data["start_loc"]) > (data["seq_length"] * 0.9), 1))
    param = new_data.filter(fun.col("same_size").between(0, 2)) \
        .groupBy("interpro_annot_accession").count().orderBy(desc("count")).limit(10)

    answer = param.collect()
    top_10 = [row.interpro_annot_accession for row in answer]
    explain = capture_explain(param)

    return [top_10, explain]


def question7_8(data):
    """
    For those annotations that also include text; in those texts:
      question 7: what are the 10 most common annotations?
      question 8: what are the 10 least common annotations?
    Param:
        data (spark.DataFrame): dataset
    Return:
        (list) : two lists with answers to the question and explain string in a list
    """
    data = data.where(data.interpro_annot_desc != "-")

    param7 = data.withColumn("word", fun.explode(fun.split(fun.col("interpro_annot_desc"), " ")))\
        .groupBy("word").count().sort("count", ascending=False).limit(10)
    param8 = data.withColumn("word", fun.explode(fun.split(fun.col("interpro_annot_desc"), " ")))\
        .groupBy("word").count().sort("count", ascending=True).limit(10)

    top_10 = param7.collect()
    least_10 = param8.collect()

    answer7 = [row.word for row in top_10]
    answer8 = [row.word for row in least_10]

    explain7 = capture_explain(param7)
    explain8 = capture_explain(param8)

    return [[answer7, explain7],[answer8, explain8]]


def question9(data):
    """
    If you combine the answers from questions 6 and 7, 
    for those largest (>90% length of the protein) features, 
    what are the top 10 common words for them?
    Param:
        data (spark.DataFrame): dataset
    Return:
        (list) : answer to the question and explain string in a list
    """
    data = data.where(data.interpro_annot_accession != "-")

    param = data.withColumn("same_size",
                            fun.when((data["stop_loc"] - data["start_loc"]) > (data["seq_length"] * 0.9), 1)) \
        .filter(fun.col("same_size").between(0, 2)) \
        .withColumn("word", fun.explode(fun.split(
            fun.col("interpro_annot_desc"), " "))) \
        .groupBy("word") \
        .count() \
        .sort("count", ascending=False) \
        .limit(10)

    top_10 = param.collect()
    answer = [row.word for row in top_10]

    explain = capture_explain(param)

    return [answer, explain]


def question10(data):
    """
    What is the coefficient (R^2) between the size of the protein itself, 
    and the number of InterPRO annotations found?
    Param:
        data (spark.DataFrame): dataset
    Return:
        (list) : answer to the question and explain string in a list
    """
    data = data.where(data.interpro_annot_accession != "-")
    param = data.select("signature_accession", "seq_length", "interpro_annot_accession") \
        .withColumn("counts",
                    fun.count("interpro_annot_accession") \
                        .over(Window.partitionBy("signature_accession"))) \
        .dropDuplicates(["signature_accession"])
    answer = param.stat.corr("seq_length", "counts")
    explain = capture_explain(param)

    return [answer, explain]


def output_writer(data):
    """
    Writes output to file
    Param:
       data (spark.DataFrame): dataset
       file (string): output csv file
    """
    with open("assignment5.csv", 'w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)

        for row in data:
            writer.writerow(row)


# Main
if __name__ == "__main__":

    dataframe = create_dataframe(sys.argv[1])

    answers = [question1(dataframe), question2(dataframe),
               question3(dataframe), question4(dataframe),
               question5(dataframe), question6(dataframe),
               question7_8(dataframe)[0], question7_8(dataframe)[1],
               question9(dataframe), question10(dataframe)]

    # Add question number
    for num, an in enumerate(answers):
        an.insert(0, num + 1)

    output_writer(answers)
