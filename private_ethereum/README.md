# A private ethereum network

A private etherum network for DAPP study and sample projects. Geth is the golang implemetation of ethereum. This network is using Geth https://geth.ethereum.org/docs/interface/private-network.

* networkid, a unique network id
* Consensus Algorithm, clique
* The Genesis Block, the first block
* Accounts, test accounts

## Setup Steps
```
geth account new --datadir /data
geth init --datadir /data /root/genesis.json
geth --datadir /data --networkid 666
```


## Test accounts
```
Public address of the key:   0x24cB6b8f255aCCefc83623b86d9Dd821bbBb0d51
Path of the secret key file: data/keystore/UTC--2021-11-15T13-26-04.165787804Z--24cb6b8f255accefc83623b86d9dd821bbbb0d51

Public address of the key:   0xdA440028eEB3D4Fe499bf51dB182595EAcCf6FFA
Path of the secret key file: data/keystore/UTC--2021-11-15T13-26-23.125925382Z--da440028eeb3d4fe499bf51db182595eaccf6ffa

Public address of the key:   0x6BdA702A32A971419268aE69F59c1441aC76A698
Path of the secret key file: /data/keystore/UTC--2021-11-15T13-33-00.507416927Z--6bda702a32a971419268ae69f59c1441ac76a698
```
