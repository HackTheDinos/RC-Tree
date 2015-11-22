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

# output tree should be:
# ((1,2),(3,4,5),6,7)