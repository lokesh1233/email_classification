# from nltk import ne_chunk, pos_tag, word_tokenize
# from nltk.tree import Tree
# import nltk
#
#
# def get_continuous_chunks(text):
#     chunked = ne_chunk(pos_tag(word_tokenize(text)))
#     continuous_chunk = []
#     current_chunk = []
#     for i in chunked:
#         if type(i) == Tree:
#             current_chunk.append(" ".join([token for token, pos in i.leaves()]))
#         elif current_chunk:
#             named_entity = " ".join(current_chunk)
#             if named_entity not in continuous_chunk:
#                 continuous_chunk.append(named_entity)
#                 current_chunk = []
#             else:
#                 continue
#     return continuous_chunk
#
# my_sent =  """
#
# Serial number : TEHP/18/12809
# Name : Arun Gadamshetty
# P No. : 153021
# Department : Spares Manufacturing Department
# Contact No : 8092084556
# Place : Vivanta by Taj - Madikeri Coorg, Coorg
# Room Type : Superior Charm
# Check In Date-Time : 10 Dec 2018 02:00 PM
# Check Out Date-Time : 12 Dec 2018 12:00 PM
# No of rooms :1
# Comments by requester : For Honeymoon trip
#
# Traveller Details:
#
# 1.
#     Name : Arun Gadamshetty
#     Relation : Self
#     Age : 34 years
# 2.
#     Name : SHWETA GADAMSHETTY
#     Relation : Spouse
#     Age : 28 years
#
# """
# #get_continuous_chunks(my_sent)
# sentence = pos_tag(word_tokenize(my_sent))
#
# grammar = "NP: {<DT>?<JJ>*<NN>}"
# cp = nltk.RegexpParser(grammar)
# result = cp.parse(sentence)
# print(result)
# sentence.draw()