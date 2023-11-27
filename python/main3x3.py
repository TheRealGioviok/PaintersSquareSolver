import numpy as np
import random
import time


def add(board, move, limit=6, inplace=True):
    res = board if inplace else board.copy()
    res += move
    res %= limit
    return res


def sub(board, move, limit=6, inplace=True):
    res = board if inplace else board.copy()
    res -= move
    res %= limit
    return res


def new_board(initialization=0, size=(3, 3)):
    return np.ones(size, dtype=np.int64) * initialization


moves = [None] * 18
movesNeg = []
# First, generate all moves
for y in range(3):
    for x in range(3):
        move = new_board()
        move[y, x] = 1
        if y > 0 and x > 0:
            move[y - 1, x - 1] = 1
        if y > 0 and x < 2:
            move[y - 1, x + 1] = 1
        if y < 2 and x > 0:
            move[y + 1, x - 1] = 1
        if y < 2 and x < 2:
            move[y + 1, x + 1] = 1
        moves[y * 3 + x] = move
        moves[9 + y * 3 + x] = -move

# Only secondary grid stuff
movesSecondary = []
movesPrimary = []
for y in range(3):
    for x in range(3):
        if (x + y) % 2 == 1:
            movesSecondary.append(moves[y * 3 + x])
        else:
            movesPrimary.append(moves[y * 3 + x])

mask_primary = new_board()
mask_secondary = new_board()

idxp = 0
idxs = 0
for y in range(3):
    for x in range(3):
        if (x + y) % 2 == 0:
            mask_primary[y, x] = 6**idxp
            idxp += 1
        else:
            mask_secondary[y, x] = 6**idxs
            idxs += 1

print(mask_primary)
print(mask_secondary)


def primary_board_id(board):
    return np.sum(board * mask_primary)


def secondary_board_id(board):
    return np.sum(board * mask_secondary)


def primary_id_to_board(id):
    board = new_board()
    idx = 0
    while id > 0:
        x = idx % 3
        y = idx // 3
        board[y, x] = id % 6
        id //= 6
        idx += 2
    return board


def secondary_id_to_board(id):
    board = new_board()
    idx = 1
    while id > 0:
        x = idx % 3
        y = idx // 3
        board[y, x] = id % 6
        id //= 6
        idx += 2
    return board


board = new_board(0)
board[0, 0] = 1

# Best-continuation (move)
arrBestP = np.zeros(6**5, dtype=np.int8)
arrTestP = np.zeros(6**5, dtype=np.int8)
arrBestP[0] = 127  # Solved

arrBestS = np.zeros(6**4, dtype=np.int8)
arrTestS = np.zeros(6**4, dtype=np.int8)
arrBestS[0] = 127  # Solved

#print(secondary_id_to_board(1296))


def encode(moven, add=True):
    return 1 + (moven + 9 * add)


def decode(enc):
    enc -= 1
    isadd = enc >= 9
    enc -= 9 * isadd
    return isadd, enc


new = True


def enlargeP(arrBest, arrNew, arrTest):
    global new
    for i in range(6**5):
        if arrBest[i] != 0 and arrTest[i] == 0:
            arrTest[i] = 1
            board = primary_id_to_board(i)
            for move in range(0, 9, 2):
                newB = add(board, moves[move], inplace=False)
                id = primary_board_id(newB)
                if arrNew[id] == 0:
                    arrNew[id] = encode(move, False)
                    new += 1
                newB = sub(newB, moves[move] * 2)
                id = primary_board_id(newB)
                if arrNew[id] == 0:
                    arrNew[id] = encode(move, True)
                    new += 1


def enlargeS(arrBest, arrNew, arrTest):
    global new
    for i in range(6**4):
        if arrBest[i] != 0 and arrTest[i] == 0:
            arrTest[i] = 1
            board = secondary_id_to_board(i)
            for move in range(1, 9, 2):
                newB = add(board, moves[move], inplace=False)
                id = secondary_board_id(newB)
                if arrNew[id] == 0:
                    arrNew[id] = encode(move, False)
                    new += 1
                newB = sub(newB, moves[move] * 2)
                id = secondary_board_id(newB)
                if arrNew[id] == 0:
                    arrNew[id] = encode(move, True)
                    new += 1


cnt = 0
tot = 0

load = False
primary = "primary3x3.npy"
secondary = "secondary3x3.npy"
if not load:
    timestart = time.time()
    while new:
        new = 0
        arrNewS = arrBestS.copy()
        enlargeS(arrBestS, arrNewS, arrTestS)
        arrBestS = arrNewS
        cnt += 1
        tot += new
        print("Run:", cnt, "- tot =", tot)
    print("Completed secondary! ({:.2f}s)".format(time.time() - timestart))

    cnt = 0
    tot = 0
    new = True
    timestart = time.time()
    while new:
        new = 0
        arrNewP = arrBestP.copy()
        enlargeP(arrBestP, arrNewP, arrTestP)
        arrBestP = arrNewP
        cnt += 1
        tot += new
        print("Run:", cnt, "- tot =", tot)

    print("Completed primary! ({:.2f}s)".format(time.time() - timestart))
    np.save("secondary3x3.npy", arrBestS)
    np.save("primary3x3.npy", arrBestP)
else:
    arrBestP = np.load(primary)
    arrBestS = np.load(secondary)


def solve3x3(board):
    pid = primary_board_id(board)
    sid = secondary_board_id(board)
    boardp = primary_id_to_board(pid)
    boards = secondary_id_to_board(sid)
    if arrBestP[pid] == 0 or arrBestS[sid] == 0:
        print("Impossible!")
        return None

    sol = new_board()
    enc = arrBestP[pid]
    while (enc != 127):
        isadd, move = decode(enc)
        x, y = move % 3, move // 3
        boardp = add(boardp, moves[move]) if isadd else sub(
            boardp, moves[move])
        pid = primary_board_id(boardp)
        sol[y, x] += -1 + (2 * isadd)
        enc = arrBestP[pid]

    enc = arrBestS[sid]
    while (enc != 127):
        isadd, move = decode(enc)
        x, y = move % 3, move // 3
        #print("Enc is {}, move is {} {}, adding = {}, sid = {}".format(enc, x, y, isadd, sid))
        boards = add(boards, moves[move]) if isadd else sub(
            boards, moves[move])
        sid = secondary_board_id(boards)
        sol[y, x] += -1 + (2 * isadd)
        enc = arrBestS[sid]

    return sol


def shuffledBoardVsOptim():
    board = new_board()
    shuffle = new_board()
    for _ in range(128):
        move = np.random.randint(9)
        x, y = move % 3, move // 3
        add(board, moves[move])
        shuffle[y, x] += 1
    shuffle %= 6
    shuffle = np.where(shuffle > 3, 6 - shuffle, shuffle)
    sol = solve3x3(board)
    return np.absolute(shuffle).sum() / np.absolute(sol).sum()



# m = .0
# for i in range(10000):
#     m += shuffledBoardVsOptim()
#     print("Curr:{:.3f}".format(m / i), end="\t\t\t\r")
# print()
