#%%
#%%
# networkx diagnosis
from statistics import mean
from turtle import color
from numpy import average, size
import pandas as pd
import json
import itertools
import csv
from operator import itemgetter
import networkx as nx
from networkx.algorithms import community #This part of networkx, for community detection, needs to be imported separately.
import numpy as np
import re
import difflib
import matplotlib.pyplot as plt

#%%
# get edges and nodes
with open('../../data/work/actor_data_weighted2.csv', 'r', encoding='utf-8') as edgecsv: # Open the file
    edgereader = csv.reader(edgecsv, delimiter='\t') # Read the csv
    #years = [e[4] for e in edgereader][1:]
    edges = [tuple(e[1:4]) for e in edgereader][1:] # Retrieve the data, columns 1 and 2
#print(edges[:5])


# the first column is the node name
node_names = [n[0] for n in edges]

with open('../../data/work/actor_data_weighted2.csv', 'r', encoding='utf-8') as edgecsv: # Open the file
    edgereader = csv.reader(edgecsv, delimiter='\t') # Read the csv
    years = [e[4] for e in edgereader][1:]
    #edges = [tuple(e[1:4]) for e in edgereader][1:]
#len(node_names)
#%%
with open('../../data/work/actor_data_weighted2.csv', 'r', encoding='utf-8') as file:
    dr = csv.DictReader(file, delimiter='\t')
    ids = []
    for e in dr:
        ids.append(e['estc_id'])
#%%
# poem filter
poem_ids = []
with open('../../data/work/s_estc_poems.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        poem_ids.append(row['id'])


#%%

years = [float(y) for y in years if y]

edges_late = []

for i,y in enumerate(years):
    if y:
        y_ = float(y)
        if y_ >= 1600: #and y_ < 1800:
            edges_late.append(edges[i])

'''for i,y in enumerate(ids):
    if y:
        y_ = y
        if y_ not in poem_ids:
            edges_late.append(edges[i])'''
# part of code above needs to be activated only if we are interested in poem network

node_names_late = [n[0] for n in edges_late]
# %%

# create graph with networkX

G = nx.Graph()

G.add_nodes_from(node_names_late)
G.add_edges_from(edges)
G.add_weighted_edges_from(edges_late)


print(nx.info(G)) # Print information about the Graph


#%%
# properties
# properties

density = nx.density(G)
print("Network density:", density)


# If your Graph has more than one component, this will return False:
print(nx.is_connected(G))

# Next, use nx.connected_components to get the list of components,
# then use the max() command to find the largest one:
components = nx.connected_components(G)
largest_component = max(components, key=len)

# Create a "subgraph" of just the largest component
# Then calculate the diameter of the subgraph, just like you did with density.
#

subgraph = G.subgraph(largest_component)
diameter = nx.diameter(subgraph)
print("Network diameter of largest component:", diameter)

#%%
# transitivity
triadic_closure = nx.transitivity(G)
print("Triadic closure:", triadic_closure)

# %%
# centrality
degree_dict = dict(G.degree(G.nodes()))
nx.set_node_attributes(G, degree_dict, 'degree')

sorted_degree = sorted(degree_dict.items(), key=itemgetter(1), reverse=True)
print("Top 10 nodes by degree:")
for d in sorted_degree[:10]:
    print(d)


betweenness_dict = nx.betweenness_centrality(G) # Run betweenness centrality
eigenvector_dict = nx.eigenvector_centrality(G) # Run eigenvector centrality

# Assign each to an attribute in your network
nx.set_node_attributes(G, betweenness_dict, 'betweenness')
nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')

sorted_betweenness = sorted(betweenness_dict.items(), key=itemgetter(1), reverse=True)

print("Top 10 nodes by betweenness centrality:")
for b in sorted_betweenness[:10]:
    print(b)

# %%
#First get the top 20 nodes by betweenness as a list
top_betweenness = sorted_betweenness[:20]

#Then find and print their degree
for tb in top_betweenness: # Loop through top_betweenness
    degree = degree_dict[tb[0]] # Use degree_dict to access a node's degree, see footnote 2
    print("Name:", tb[0], "| Betweenness Centrality:", tb[1], "| Degree:", degree)


# community detection

communities = community.greedy_modularity_communities(G)

modularity_dict = {} # Create a blank dictionary
for i,c in enumerate(communities): # Loop through the list of communities, keeping track of the number for the community
    for name in c: # Loop through each person in a community
        modularity_dict[name] = i # Create an entry in the dictionary for the person, where the value is which group they belong to.

# Now you can add modularity information like we did the other metrics
nx.set_node_attributes(G, modularity_dict, 'modularity')

# First get a list of just the nodes in that class
class0 = [n for n in G.nodes() if G.nodes[n]['modularity'] == 0]

# Then create a dictionary of the eigenvector centralities of those nodes
class0_eigenvector = {n:G.nodes[n]['eigenvector'] for n in class0}

# Then sort that dictionary and print the first 5 results
class0_sorted_by_eigenvector = sorted(class0_eigenvector.items(), key=itemgetter(1), reverse=True)

print("Modularity Class 0 Sorted by Eigenvector Centrality:")
for node in class0_sorted_by_eigenvector[:5]:
    print("Name:", node[0], "| Eigenvector Centrality:", node[1])

#%%
#draw

import matplotlib.pyplot as plt


nx.draw(G, node_size=2, width=1, edge_color="black")


# %%
# make dict actors - roles
act_rol = {}
with open('../../data/work/s_estc_full.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row['actor_id'] not in act_rol:
            act_rol[row['actor_id']] =  {row['actor_roles_all']: 1}
        else:
            if row['actor_roles_all'] not in act_rol[row['actor_id']]:
                act_rol[row['actor_id']][row['actor_roles_all']] = 1
            else:
                act_rol[row['actor_id']][row['actor_roles_all']] += 1


for e in act_rol:
    act_rol[e] = max(act_rol[e], key=act_rol[e].get)

print(act_rol)
#%%
names_act_rol = {}
with open('../../data/work/s_estc_full.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row['name_unified'] not in names_act_rol:
            names_act_rol[row['name_unified']] =  {row['actor_roles_all']: 1}
        else:
            if row['actor_roles_all'] not in names_act_rol[row['name_unified']]:
                names_act_rol[row['name_unified']][row['actor_roles_all']] = 1
            else:
                names_act_rol[row['name_unified']][row['actor_roles_all']] += 1


for e in names_act_rol:
    names_act_rol[e] = max(names_act_rol[e], key=names_act_rol[e].get)

print(names_act_rol)
#%%
# show actors with top betweenness
# (the same we can do for degree and eigenvector)
for e in sorted_betweenness[:100]:
    try: print(act_rol[e[0]])
    except:
        try:
            '''
        try:
            name = re.sub(r' \((.*?)\)', r', \1.', e[0])
            print(names_act_rol[name])
        except: print(name)
        '''
            name = difflib.get_close_matches(e[0], names_act_rol.keys())[0]
            print(names_act_rol[name])
        except: print(e)
#%%
id_to_names = {}
with open('../../data/work/s_estc_full.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        id_to_names[row['actor_id']] =  row['name_unified']
print(id_to_names)
#%%
for e in sorted_betweenness[:100]:
    try: print(id_to_names[e[0]])
    except: print(e[0])
#%%
roles = []
for e in sorted_betweenness:
    try: roles.append(act_rol[e[0]])
    except:
        try:
            name = difflib.get_close_matches(e[0], names_act_rol.keys())[0]
            roles.append(names_act_rol[name])
        except: pass

print(len(roles))
#%%
# print plots with percentage of roles 
dict_roles = {i:roles[:100].count(i) for i in roles[:100]} # change ":100" to just ":" if you want a plot for all actors, not just the top 100

sm = sum(dict_roles.values())
for e in dict_roles.keys():
    dict_roles[e] = round(100*dict_roles[e] / sm, 2)
print(dict_roles)


dict_roles = {'publisher': dict_roles['publisher'], 'bookseller': dict_roles['bookseller'],
'printer': dict_roles['printer'], 'author': dict_roles['author']}

dict_roles['others'] = 100 - sum(dict_roles.values())

plt.bar(range(len(dict_roles)), list(dict_roles.values()), align='center')
plt.xticks(range(len(dict_roles)), list(dict_roles.keys()))
plt.show()

#%%
# counting mediana for years

yrs = []
with open('../../data/work/s_estc_full.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        try: yrs.append(int(row['publication_year'][:4]))
        except: pass
print(np.median(yrs))
print(len(yrs))