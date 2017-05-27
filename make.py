#!/usr/bin/python
import argparse
import os
import re
import subprocess
from pathlib import Path


def get_classpath():
	''' Retrieve Java executable classpath '''
	classpath = Path('.classpath')

	if classpath.is_file():
		# read classpath if already saved
		with open(str(classpath), 'r') as classpath_file:
			return str(classpath_file.read().strip())
	else:
		# create it if not available
		classpath_elements = subprocess.check_output(
			'./gradlew showDepsClasspath | grep jar',
			shell=True).strip().decode('utf-8')
		classpath_elements += ':build/classes/main'

		with open(str(classpath), 'w') as classpath_file:
			classpath_file.write(classpath_elements)

		return classpath_elements

def exec_command(command, **env_variables):
	''' Execute command via bash with given environment variables '''
	# convert path to string representation to feed subprocess.call
	if isinstance(command, Path):
		command = str(command)

	#print('------> ', command)
	subprocess.call(command,
		shell=True,
		# load all environment variable and add new wanted ones
		env={**env_variables, **dict(os.environ)})

def master_link(link):
	''' Validate proper Spark master links '''
	if link.startswith('local') or link.startswith('spark://'):
		return link
	raise ValueError("Invalid Spark master link: " + link)

def main_class(class_name, default_path='it.unipd.dei.dm1617.examples.'):
	''' Validate class name, adding default_path if needed '''
	if class_name.startswith('it.'):
		return
	else:
		return default_path + class_name

if __name__ == '__main__':
	# parse command line arguments into cmd_args object
	exec_command('clear')
	parser = argparse.ArgumentParser(description='Manage Spark runs')

	parser.add_argument('arguments',
		nargs='*',
		help='Args for Java main')
	parser.add_argument('--master',
		dest='master',
		type=master_link, # function to validate argument
		default='local',
		help='link for wanted Spark master')
	parser.add_argument('--spark-path',
		dest='spark_path',
		type=Path,
		default='/opt/apache-spark',
		help='Directory Spark is installed on the system (for distributed runs)')
	parser.add_argument('--class',
		dest='main_class',
		type=main_class,
		help='Class containing main static method to run')

	cmd_args = parser.parse_args()

	# compile project
	exec_command('./gradlew compileJava')

	# create a project jar (needed for parallel execution) if not present
	if not Path('build/libs/data_mining_project-1.0-SNAPSHOT-all.jar').is_file():
		exec_command('./gradlew shadowJar')

	# restart all spark instances if running distributed
	if not cmd_args.master.startswith('local'):
		exec_command(cmd_args.spark_path.joinpath('sbin', 'stop-all.sh'))
		exec_command(cmd_args.spark_path.joinpath('sbin', 'start-all.sh'),
			# path for native hadoop libraries
			LD_LIBRARY_PATH='/usr/lib/hadoop/lib/native/:$LD_LIBRARY_PATH')

	# run chosen class
	command = 'java -Dspark.master={} -cp $CP {main_class} {cmd_args}'\
		.format(cmd_args.master,
			main_class=cmd_args.main_class,
			cmd_args=' '.join(cmd_args.arguments))
	exec_command(command,
		CP=get_classpath(),
		# path for native hadoop libraries
		LD_LIBRARY_PATH='/usr/lib/hadoop/lib/native/:$LD_LIBRARY_PATH')
