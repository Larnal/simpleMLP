import matplotlib.pyplot as plt
from neuralnetwork import *
import time

print("lo")
# donnees a apprendre:
x = normalize(np.load('basetrain.npy'))
labels = np.load('labeltrain.npy')
xtest = normalize(np.load('basetest.npy'))
labeltest = np.load('labeltest.npy')

target = label2target(labels)
testtarget = label2target(labeltest)

# parametres d’apprentissage:
c = 16  # Nombre de perceptrons couche cachée
w1, w2 = mlp2def(np.size(x, 0), c, np.size(target, 0))  # pour MLP 2 couches
w = mlp1def(np.size(x, 0), np.size(target, 0))  # pour MLP 1 couche
it_train = 10
epoch = 10
lr = 0.005
qerror = np.empty([1], dtype=float)

start = time.time()  # Start time measurement

# Entrainement pour MLP 1 couche
# print("Apprentissage MLP1 ...")
# Lscore = np.asarray([list(score(mlpclass(mlp1run(x, w)), labels))])
# Gscore = np.asarray([list(score(mlpclass(mlp1run(xtest, w)), labeltest))])
# for i in range(epoch):
#     # apprentissage:
#     print("epoch: ", i)
#     w, L = mlp1train(x, target, w, lr, it_train)
#     qerror = np.append(qerror, L)
#     Lscore = np.append(Lscore, [list(score(mlpclass(mlp1run(x, w)), labels))], axis=0)
#     Gscore = np.append(Gscore, [list(score(mlpclass(mlp1run(xtest, w)), labeltest))], axis=0)


# Entrainement pour MLP 2 couches
print("Apprentissage MLP2 ...")
Lscore = np.asarray([list(score(mlpclass(mlp2run(x, w1, w2)), labels))])
Gscore = np.asarray([list(score(mlpclass(mlp2run(xtest, w1, w2)), labeltest))])
for i in range(epoch):
    # apprentissage:
    print("epoch: ", i)
    w1, w2, L = mlp2train(x, target, w1, w2, lr, it_train)
    qerror = np.append(qerror, L)
    Lscore = np.append(Lscore, [list(score(mlpclass(mlp2run(x, w1, w2)), labels))], axis=0)
    Gscore = np.append(Gscore, [list(score(mlpclass(mlp2run(xtest, w1, w2)), labeltest))], axis=0)

time = time.time() - start

# affichage erreur quadratique:
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

# affichage Taux d'erreur:
plt.plot(axe_x, Lscore[:, 1], 'r--', axe_x, Gscore[:, 1], 'b--')
plt.title('Taux d\'apprentissage')
plt.xlabel("%")
plt.xlabel("Iterations de test")
plt.legend(('Apprentissage', 'Généralisation'),
           loc='lower right')
plt.show()
print("finished in ", time, "s !")
