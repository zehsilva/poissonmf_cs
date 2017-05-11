import numpy as np
import pandas as pd
import os

class LoadModify(object):
    def __init__(self,folder='/home/eliezer/datasets/hetrec2011/lastfm/',save=None):
        self.save=save;
        self.rootfolder = folder
        
        self.artists_id = None
        self.users_id = None
        self.artists_inv_id = None
        self.users_inv_id = None
        
        self.list_friends_id = None
        self.tags_id = None
        self.tags_inv_id = None
        self.map_tag_id_name = None
        self.map_tag_name_id = None
        self.artists_tags_id = None
        
        self.n_tags = None
        self.n_artists = None
        self.n_max_friends = None
        self.n_users = None
        
        self.mat_artists_tags = None
        self.mat_users_friends = None
        self.mat_users_artists = None

        self.mat_users_artists_train = None
        self.mat_users_artists_test = None
        self.userid_nitems=None

        self.train_percentage = 0.90
        self.min_train_items = 5
        
        self.file_user_item = "user_artists.dat"
        self.file_item_tags = "user_taggedartists.dat"
        self.file_user_user = "user_friends.dat"
        self.file_tags = "tags.dat"
        
        self.newfile_user_item = None
        self.newfile_item_tags = None
        self.newfile_user_user = None
        self.newfile_tags = None
    def load_save(self,train_percentage=0.8):
        if self.save==None:
            self.save="p"+str(int(train_percentage*100))+"_train_test_"+str(np.random.randint(10000))+'/'
        try: 
            os.makedirs(self.rootfolder +self.save)
        except OSError:
            if not os.path.isdir(self.rootfolder +self.save):
                raise
        self.load(train_percentage)
        W = self.mat_artists_tags
        S = self.list_friends_id
        self.datafolder = self.rootfolder +self.save
        print "saving to",self.datafolder
        np.savetxt(self.datafolder+"user_artist_rating.train",self.list_train,fmt='%i',delimiter='\t')
        np.savetxt(self.datafolder+"user_artist_rating.test",self.list_test,fmt='%i',delimiter='\t')

        new_list_friends_id=np.array([ (i, item_i)  for i, item in enumerate(self.list_friends_id) for item_i in item])
        np.savetxt(self.datafolder+"v1_user_friends.dat",new_list_friends_id,delimiter='\t',fmt='%i')

        W = np.array(self.mat_artists_tags,dtype=int)
        np.savetxt(self.datafolder+"tag_artist_count.dat",W,fmt='%i',delimiter='\t')

        # keep a list with tags and index
        tags_id_name=np.array(zip([self.tags_inv_id[x] for x in\
                                   self.tags_id],self.generate_artists_tags(self.tags_id)))
        pd.DataFrame(tags_id_name[:,1],index=tags_id_name[:,0],columns=['tag'])\
        .to_csv(self.datafolder+"tag_id_name.dat",sep='\t',header=False)

    def load(self,train_percentage=0.80):
        self.train_percentage=train_percentage
        print "train percentage = ",100*self.train_percentage
        # load user-artist list and generate list of user and artist id
        userartists = np.loadtxt(self.rootfolder+"user_artists.dat", delimiter='\t',skiprows=1,dtype=int)
        #userartists=self.filter_user_items(userartists,0.7) #randomly filter out some items
        self.userartists=userartists
        self.artists_id = np.unique(userartists[:,1])
        self.users_id = np.unique(userartists[:,0])
        #self.user_artist_tuple = userartists
        self.userid_nitems = np.array([[idx, len(userartists[userartists[:, 0] == idx])] for idx in self.users_id])


        self.artists_inv_id = dict(zip(self.artists_id, xrange(len(self.artists_id))))
        self.users_inv_id = dict(zip(self.users_id, xrange(len(self.users_id))))

        
        # load user friends and generate list of friends
        userfriends = np.loadtxt(self.rootfolder+"user_friends.dat", delimiter='\t',skiprows=1,dtype=int)
        # this commented version the user_id is not the index in the array, but the actual id, needs the index
        # for further computation
        # self.list_friends_id= [ userfriends[userfriends[:,0]==u_id][:,1] for u_id in users_id]
        self.list_friends_id = [[self.users_inv_id[friend_id] for friend_id in \
                            userfriends[userfriends[:, 0] == u_id][:, 1] if friend_id in self.users_id] for u_id in self.users_id]
        del userfriends

        # load tags and generates mapping from tag text to tag id and tag_id to index in array.
        tags = pd.read_csv(self.rootfolder+ "tags.dat", sep='\t')
        self.tags_id = tags['tagID'].get_values()
        self.tags_inv_id = dict(zip(self.tags_id, xrange(len(self.tags_id))))
        self.map_tag_id_name = dict(zip(self.tags_id, map(str, tags['tagValue'].get_values())))
        self.map_tag_name_id = dict(zip(map(str, tags['tagValue'].get_values()), self.tags_id))
        del tags

        # load user-artists-tags and generate list of tags per artist
        ua_tag = np.loadtxt(self.rootfolder+"user_taggedartists.dat",delimiter='\t',skiprows=1,dtype=int)
        self.artists_tags_id = [ [self.tags_inv_id[a_tag] for a_tag in   ua_tag[ua_tag[:,1]==a_id][:,2]] for a_id in self.artists_id ]
        
        self.n_tags = self.tags_id.size
        self.n_artists = self.artists_id.size
        self.n_max_friends = np.max(np.array([len(x) for x in self.list_friends_id]))
        self.n_users = self.users_id.size
        self.make_tag_matrix()
        #self.make_user_item_matrix(userartists)
        self.make_train_test_matrix(userartists)
        del userartists

    def generate_artists_tags(self,lst_tags_id):
        return [val.replace("-","") for x in lst_tags_id for val in self.map_tag_id_name[x].split()]

    def make_tag_matrix(self):
        self.mat_artists_tags = []
        for pos,artist_tags in enumerate(self.artists_tags_id):
            count=count_to_array( artist_tags, self.n_tags )
            #print count
            self.mat_artists_tags+=[[pos,tag_idx,tagc] for tag_idx,tagc in enumerate(count) if tagc>0]
    
    def make_user_item_matrix(self,user_artist_tuple):
        self.mat_users_artists = np.zeros ( shape = (self.n_users,self.n_artists))
        print "user_item_matrix"
        users_indexes=np.fromiter([self.users_inv_id[xi] for xi in user_artist_tuple[:,0]],user_artist_tuple[:,0].dtype)
        artist_indexes=np.fromiter([self.artists_inv_id[xi] for xi in user_artist_tuple[:,1]],user_artist_tuple[:,1].dtype)
        self.mat_users_artists[users_indexes,artist_indexes]=1

    def n_train_per_user(self,user_id):
        # return the number of training items that should be selected for the user user_id
        return int(np.floor(col2wcol1(self.userid_nitems, user_id).flatten()[0] * self.train_percentage))

    def filter_users(self,min_n_items):
        # guarantee that all users in the training set will have more or equal to min_n_items items
        return self.userid_nitems[np.floor(self.train_percentage * self.userid_nitems[:, 1]) > min_n_items][:, 0]

    def training_idxs(self,user_artist_tuple,user_id):
        return np.sort(np.random.choice(user_row_idx(user_artist_tuple, user_id), self.n_train_per_user(user_id), replace=False))

    def make_train_test_matrix(self, user_artist_tuple):
        self.mat_users_artists_train = np.zeros(shape=(self.n_users, self.n_artists))
        self.mat_users_artists_test = np.zeros(shape=(self.n_users, self.n_artists))

        print "sampling training indices"
        user_filter_train_index = np.hstack(tuple((self.training_idxs(user_artist_tuple,user_i) for user_i in self.filter_users(self.min_train_items))))

        print "max_train_idx=",np.max(user_filter_train_index)
        print "calculating testing indices"
        user_filter_test_index = np.setdiff1d(np.array(range(user_artist_tuple.shape[0])),user_filter_train_index)
        print "max_test_idx=", np.max(user_filter_test_index)
        users_indexes=np.fromiter([self.users_inv_id[xi] for xi in user_artist_tuple[:,0]],user_artist_tuple[:,0].dtype)
        artist_indexes=np.fromiter([self.artists_inv_id[xi] for xi in user_artist_tuple[:,1]],user_artist_tuple[:,1].dtype)

        print "build dense matrixes"
        users_indexes_train = users_indexes[user_filter_train_index]
        artist_indexes_train = artist_indexes[user_filter_train_index]
        self.list_train=np.vstack((users_indexes_train,artist_indexes_train ,user_artist_tuple[user_filter_train_index,2])).T
        users_indexes_test = users_indexes[user_filter_test_index]
        artist_indexes_test = artist_indexes[user_filter_test_index]
        self.list_test=np.vstack((users_indexes_test,artist_indexes_test ,user_artist_tuple[user_filter_test_index,2])).T
        self.mat_users_artists_train[users_indexes_train, artist_indexes_train] = user_artist_tuple[user_filter_train_index,2]
        self.mat_users_artists_test[users_indexes_test, artist_indexes_test] = user_artist_tuple[user_filter_test_index,2]
        
        
# select index row index of a user given user index in column 0
def user_row_idx(user_artist_tuple, user_id):
    return np.where(user_artist_tuple[:, 0] == user_id)[0]


# select column 2 using column 1 as index
def col2wcol1(twocolumns,idxcol1):
    return twocolumns[twocolumns[:, 0] == idxcol1][:,1]


# select the column 1 using column 2 as index
def col1wcol2(twocolumns,idxcol2):
    return twocolumns[twocolumns[:, 1] == idxcol2][:,0]


class Counter(object):
    def __init__(self):
        self.counter=dict()
        
    def add(self,x):
        if self.counter.has_key(x):
            self.counter[x]+=1
        else:
            self.counter[x]=1
        return self
    
    def add_all(self,lst):
        for x in lst:
            self.add(x)
        return self
    
    def to_array(self,size):
        new_array=np.zeros(shape=(size,))
        for key,val in self.counter.iteritems():
            if key >= size:
                raise AssertionError("key ("+str(key)+") >= size ("+str(size)+")")
            new_array[key]=val
        return new_array
    
    def to_list(self):
        new_lst=[]
        for key,val in self.counter.iteritems():
            if key >= size:
                raise AssertionError("key ("+str(key)+") >= size ("+str(size)+")")
            new_lst.append((key,val))
        return new_lst


def count_to_array(list_items,max_size):
    return Counter().add_all(list_items).to_array(max_size)

def count_to_list(list_items):
    return Counter().add_all(list_items).to_array()

