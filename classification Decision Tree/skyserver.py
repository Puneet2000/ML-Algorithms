import csv
file = csv.reader(open('skyserver.csv','r'),delimiter=',',quoting=csv.QUOTE_NONNUMERIC)

TRAINING_SIZE = 1000
def gini_index(groups,classes):
	#print("calculating ginni")
	n_instances = float(sum([len(group) for group in groups]))
	group = 0.0

	for group in groups:
		size = len(group)
		if size == 0:
			continue
		score =0.0
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p*p

		gini = (1 - score)*(size / n_instances)

	return gini

def test_split(index , value , dataset):
	#print("test_split()")
	left , right = list() , list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else :
			right.append(row)
	return left , right

def get_split(dataset):
	print("get_split()")
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

def to_terminal(group):
	#print("to_terminal()")
	outcomes = [row[-1] for row in group]
	return max(set(outcomes) , key = outcomes.count)

def split(node, max_depth , min_size,depth):
	print("split()")
	left, right = node['groups']
	del(node['groups'])

	if not left or not right:
		node['left'] = node['right'] = to_terminal(left+right)
		return

	if depth>=max_depth:
		node['left'] , node['right'] = to_terminal(left) , to_terminal(right)
		return

	if len(left) <=min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'] , max_depth,min_size,depth+1)

	if len(right) <=min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'] , max_depth,min_size,depth+1)

	return

def build_tree(train , max_depth , min_size):
	#print("build_tree()")
	root = get_split(train)
	split(root,max_depth,min_size,1)
	return root

def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))

def predict(node,row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'],dict):
			return predict(node['left'],row)
		else:
			return node['left']

	else:
		if isinstance(node['right'],dict):
			return predict(node['right'],row)
		else:
			return node['right']

dataset = list()
test = list()
length =0
for row in file:
	length = length+1
	if length > TRAINING_SIZE and length <2000:
		test.append(row)
	elif length >=2000:
		break;
	dataset.append(row)


#print(dataset)
tree = build_tree(dataset,8,10)
#print_tree(tree)
passed =0
for row in test:
	if predict(tree,row) == row[-1]:
		passed = passed+1

print(passed)
