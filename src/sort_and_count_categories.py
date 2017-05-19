#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os.path

ARTICLES_ASSIGNED_THRESHOLD = 2
INPUT_FILE = "../output/sort_and_count_categories/part-"
PRINT_TOP_CATEGORIES = False

# you can pass as second argument the number of articles a category have been assigned to 
if len(sys.argv) >= 2:
	ARTICLES_ASSIGNED_THRESHOLD = int(sys.argv[1])
else:
	print("You can pass the thresold of articles for categories as second argument of the script\n")

if( not(os.path.isfile(INPUT_FILE + "00000")) ):
	print("INPUT NOT FOUND! Run countCategories.java (or check output/sort_and_count_categories)")
	exit(1)

report = []
part_file_counter = 0
# read all the categories and their count
while os.path.isfile(INPUT_FILE + str(part_file_counter).zfill(5)):
	with open(INPUT_FILE + str(part_file_counter).zfill(5)) as report_file:
		for line in report_file:
			# remove parenthesis
			tup = line.strip()[1:-1].split(",")
			# name of categories may contain commas, lets join it back
			category = ",".join(tup[:-1])
			n_articles = tup[-1]
			report.append( (category, int(n_articles)) )
	part_file_counter += 1

# sort by number of articles for category
res = sorted(report, key=lambda tup: tup[1])

# we have one category per line
print("Total number of categories (medium dataset): " + str(len(res)) + "\n")

if PRINT_TOP_CATEGORIES:
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