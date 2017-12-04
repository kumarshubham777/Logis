__author__ = 'Shubham'
import numpy as np
import galg as NN
import random
from operator import itemgetter, attrgetter, methodcaller


#def genetic():
mutation_rate=0.05
crossover_rate=0.7
popN=100
def iteratePop (curPop,fitVec):
  nextPop=[]
  for i in range(20):
      nextPop.append(curPop[i])
  while len(nextPop) != popN:
    ch1, ch2 = [], []
    ch1, ch2 = selectFittest (fitVec, curPop) # select two of the fittest chromos

    ch1, ch2 = breed (ch1, ch2) # breed them to create two new chromosomes
    nextPop.append(ch1) # and append to new population
    nextPop.append(ch2)
  return nextPop
def selectFittest (fitVec, curPop):
  while 1 == 1: # ensure that the chromosomes selected for breeding are have different indexes in the population
    index1 = roulette (fitVec)
    index2 = roulette (fitVec)
    if index1 == index2:
      continue
    else:
      break


  ch1 = curPop[int(fitVec[index1][0])] # select  and return chromosomes for breeding
  ch2 = curPop[int(fitVec[index2][0])]
  return ch1, ch2

"""Fitness scores are fractions, their sum = 1. Fitter chromosomes have a larger fraction.  """
def roulette (fitVec):
    '''
  index = 0
  cumalativeFitness = 0.0
  r = random.random()

  for i in range(len(fitVec)): # for each chromosome's fitness score
    cumalativeFitness += fitVec[i][1] # add each chromosome's fitness score to cumalative fitness

    if cumalativeFitness > r: # in the event of cumalative fitness becoming greater than r, return index of that chromo
      return int(fitVec[i][0]-1)
      '''
    best=-1
    for i in range(2):
        r=random.randint(0,99)
        if ((best==-1)or(fitVec[r][1]>fitVec[best][1])):
            best=r
    return best
def breed (ch1, ch2):

  newCh1, newCh2 = [], []
  if random.random() < crossover_rate: # rate dependent crossover of selected chromosomes
   newCh1, newCh2 = crossover(ch1, ch2)
  else:
   newCh1, newCh2 = ch1, ch2
  newnewCh1 = mutate (newCh1) # mutate crossovered chromos
  newnewCh2 = mutate (newCh2)

  return newnewCh1, newnewCh2
def crossover (ch1, ch2):
  r=random.randint(0,17)
  list1=[]
  list2=[]
  for i in range(r):
      list1.append(ch1[i])
      list2.append(ch2[i])
  for j in range(17-r+1):
      list1.append(ch2[j+r])
      list2.append(ch1[j+r])
  return list1,list2




def mutate (ch):
  mutatedCh = []
  for i in ch:
    if random.random() < mutation_rate:
      if i == 1:
        mutatedCh.append(0)
      else:
        mutatedCh.append(1)
    else:
      mutatedCh.append(i)
  #assert mutatedCh != ch
  return mutatedCh



params = [100, 0.05, 100, 18,20]
curPop=np.random.randint(2, size=(100,18))
nextPop = np.zeros((curPop.shape[0], curPop.shape[1]))
fitVec = np.zeros((100, 2))
#3def genetic():
for i in range(params[0]):
 for e in range(100):
  fitVec[e][1]=NN.LR(curPop[e])
  fitVec[e][0]=e
 fitVec=sorted(fitVec,key=itemgetter(1),reverse=True)
 nextPop=iteratePop(curPop,fitVec)
 curPop=nextPop

print(fitVec[0][1])
print(curPop[int(fitVec[0][0])])






