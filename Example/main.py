import Cobra


MySearch = Cobra.Search()

#MySearch.addDatFile('NoBinary')
MySearch.addDatFile('Ter5A_test')

MySearch.addCandidate('Ter5A_prior.dat')
MySearch.ChainRoot = './results/Ter5A_test-'

MySearch.sample(doplot = True, resume=True, nlive = 200)



