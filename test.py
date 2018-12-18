import matplotlib.pyplot as plt
from neuralnetwork import *
import time

""" ########################  Données ##################################"""
x = np.load('basetrain.npy')
labels = np.load('labeltrain.npy')
xtest = np.load('basetest.npy')
labeltest = np.load('labeltest.npy')

target = label2target(labels)
""" ########################  Parametres d’apprentissage ###############"""


n = np.size(x, 0)                                           # Nb element dans sample
m = np.size(target, 0)                                      # Nb de classe
c = 80                                                      # Nombre de perceptrons couche cachée

it_train = 2                                                # Nb d'itérations d'apprentissage (dans un cycle)
epoch = 50                                                  # Nb d'epoch (test sur base de test)
lr = 0.05                                                   # Learning rate


qerror = np.empty([1], dtype=float)                         # vecteur d'erreur quadratique
start = time.time()                                         # Mesure du temps d'exécution

""" ########################  Entrainement MLP 1 couche ###############"""
# print("Apprentissage MLP1 ...")
# w = mlp1def(np.size(x, 0), np.size(target, 0))  # pour MLP 1 couche
# Lscore = np.asarray([list(score(mlpclass(mlp1run(x, w)), labels))])
# Gscore = np.asarray([list(score(mlpclass(mlp1run(xtest, w)), labeltest))])
# for i in range(epoch):
#     # apprentissage:
#     print("epoch: ", i)
#     w, L = mlp1train(x, target, w, lr, it_train)
#     qerror = np.append(qerror, L)
#     Lscore = np.append(Lscore, [list(score(mlpclass(mlp1run(x, w)), labels))], axis=0)
#     Gscore = np.append(Gscore, [list(score(mlpclass(mlp1run(xtest, w)), labeltest))], axis=0)


""" ########################  Entrainement MLP 2 couches (1 cachée) ###############"""
print("Apprentissage MLP2 ...")
w1, w2 = mlp2def(np.size(x, 0), c, np.size(target, 0))      # pour MLP 2 couches
Lscore = np.asarray([list(score(mlpclass(mlp2run(x, w1, w2)), labels))])
Gscore = np.asarray([list(score(mlpclass(mlp2run(xtest, w1, w2)), labeltest))])
for i in range(epoch):
    # apprentissage:
    print("epoch: ", i)
    w1, w2, L = mlp2train(x, target, w1, w2, lr, it_train)
    qerror = np.append(qerror, L)
    Lscore = np.append(Lscore, [list(score(mlpclass(mlp2run(x, w1, w2)), labels))], axis=0)
    Gscore = np.append(Gscore, [list(score(mlpclass(mlp2run(xtest, w1, w2)), labeltest))], axis=0)

""" ########################  FIN SECTION ENTRAINEMENT ###############"""



time = time.time() - start

""" ########################  Affichage ###############################"""
# affichage erreur quadratique
axe_x = np.linspace(1, it_train * epoch, it_train * epoch)
plt.plot(axe_x, qerror[1:])
plt.ylabel('Cout quadratique')
plt.xlabel("Iterations d’apprentissage")
plt.title("Cout quadratique")
plt.show()

# affichage Scores:
axe_x = np.linspace(1, epoch + 1, epoch + 1)
plt.plot(axe_x, Lscore[:, 0], 'r', axe_x, Gscore[:, 0], 'b')
plt.title('Score')
plt.xlabel("Score")
plt.xlabel("Iterations de test")
plt.legend(('Score d\'apprentissage', 'Score de généralisation'),
           loc='lower right')
plt.show()

# affichage Taux de réussite / apprentissage:
plt.plot(axe_x, Lscore[:, 1], 'r--', axe_x, Gscore[:, 1], 'b--')
plt.title('Taux d\'apprentissage')
plt.xlabel("%")
plt.xlabel("Iterations de test")
plt.legend(('Apprentissage', 'Généralisation'),
           loc='lower right')
plt.show()
print("exécuté en ", round(time, 0), "s !")
print("Taux de reussite sur base d'entrainement = ", Lscore[-1,1])
print("Taux de reussite sur base de test = ", Gscore[-1,1])

