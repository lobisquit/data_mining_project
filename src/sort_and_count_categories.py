#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os.path

ARTICLES_ASSIGNED_THRESHOLD = 2
INPUT_FILE = "../output/sort_and_count_categories/sort_and_count_categories"

# you can pass as second argument the number of articles a category have been assigned to 
if len(sys.argv) >= 2:
	ARTICLES_ASSIGNED_THRESHOLD = int(sys.argv[1])
else:
	print("You can pass the thresold of articles for categories as second argument of the script\n")

if( not(os.path.isfile(INPUT_FILE)) ):
	print("INPUT NOT FOUND! This script needs as input the sort_and_count_categories file (available on dropbox). \n Put that file into "+INPUT_FILE)
	exit(1)

report = []
# read all the categories and their count
# you can find the precomputed file (relative to the medium dataset) sort_and_count_categories in the dropbox folder
with open(INPUT_FILE) as report_file:
	for line in report_file:
		# remove parenthesis
		tup = line.strip()[1:-1].split(",")
		# name of categories may contain commas, lets join it back
		category = ",".join(tup[:-1])
		n_articles = tup[-1]
		report.append( (category, int(n_articles)) )

# sort by number of articles for category
res = sorted(report, key=lambda tup: tup[1])

# we have one category per line
print("Total number of categories (medium dataset): " + str(len(res)) + "\n")

i = 0
for tup in res:
	if tup[1] >= 100:
		words = tup[0].split(" ")

		# lets exclude the "births 19xx" and "deaths 19xx" categories
		if not(len(words) >= 2):
			secondWord = "does not matter as long it is not births or deaths"
		else:	
			secondWord = words[1]
		if not(secondWord == "births" or secondWord == "deaths"):
			i +=1
			print(tup)
print("Categories with 100 or more articles (excluding births year and deaths year): " + str(i) + "\n")


i = 0
for tup in res:
	if tup[1] >= ARTICLES_ASSIGNED_THRESHOLD:
		i +=1
print("Categories with " + str(ARTICLES_ASSIGNED_THRESHOLD) + " or more articles: " + str(i))