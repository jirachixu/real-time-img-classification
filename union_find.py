class union_find:
    def __init__(self):
        self.parent = {}
        self.size = {}
    
    def make_set(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1
            
    def find_set(self, x):
        if x == self.parent[x]:
            return x
        # Path compression, will set the parent of x directly to the root
        self.parent[x] = self.find_set(self.parent[x])
        return self.parent[x]
    
    def union_sets(self, x, y):
        root_x = self.find_set(x)
        root_y = self.find_set(y)
        if root_x != root_y:
            if self.size[root_x] < self.size[root_y]:
                root_x, root_y = root_y, root_x
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]