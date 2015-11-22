import tree2
import ete2

# test
labels = {
        'a':7, 
        'b':None,
        'c':6,
        'd':None,
        'e':None,
        'f':5,
        'g':1,
        'h':2,
        'i':3,
        'j':4
    }
edges = [
        ('a', 'b'), 
        ('b', 'c'),
        ('b', 'd'),
        ('b', 'e'),
        ('e', 'f'),
        ('j', 'e'),
        ('e', 'i'),
        ('g', 'd'),
        ('h', 'd')
]


tree1 = ete2.Tree("{};".format(((1,2),(3,4,5),6,7)), format=1)
tree2 = ete2.Tree("{};".format(tree2.build_tree(labels, edges)), format=1)

assert(tree1.robinson_foulds(tree2, unrooted_trees=True)[0] == 0)
print "tree2.build_tree working"