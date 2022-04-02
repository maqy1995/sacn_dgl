# SACN

Paper: "[End-to-end Structure-Aware Convolutional Networks for Knowledge Base Completion](https://arxiv.org/pdf/1811.04441.pdf)" 

original code:"https://github.com/JD-AI-Research-Silicon-Valley/SACN"

Excluding `spodernet` and implementing WGCN using DGL.

Dependency:
> dgl 0.5


FB15K-237:
params:
```
/home/maqy/miniconda3/envs/work/bin/python /home/maqy/work/sacn_dgl/sacn_dgl.py --dataset FB15k-237
Using backend: pytorch
Namespace(batch_size=128, channels=200, dataset='FB15k-237', dropout_rate=0.2, embedding_dim=200, eval_every=1, gc1_emb_size=150, gpu=0, init_emb_size=100, input_dropout=0, kernel_size=5, lr=0.002, n_epochs=5000, num_workers=2, patience=100)
# entities: 14541
# relations: 237
# training edges: 272115
# validation edges: 17535
# testing edges: 20466
Done loading data from cached files.
14541 475
Graph(num_nodes=14541, num_edges=558771,
      ndata_schemes={}
      edata_schemes={})
```
valid:
```
epoch : 453
epoch time: 57.6528
loss: 0.00129879848100245

--------------------------------------------------
dev_evaluation
--------------------------------------------------

Hits left @1: 0.3589392643284859
Hits right @1: 0.16230396350156828
Hits @1: 0.2606216139150271
Hits left @2: 0.4465925292272598
Hits right @2: 0.2275449101796407
Hits @2: 0.33706871970345026
Hits left @3: 0.4947248360422013
Hits right @3: 0.2699743370402053
Hits @3: 0.3823495865412033
Hits left @4: 0.5271742229826062
Hits right @4: 0.30287995437696036
Hits @4: 0.4150270886797833
Hits left @5: 0.5556315939549472
Hits right @5: 0.33179355574565156
Hits @5: 0.4437125748502994
Hits left @6: 0.5773595665811234
Hits right @6: 0.35483319076133446
Hits @6: 0.466096378671229
Hits left @7: 0.5938979184488167
Hits right @7: 0.3737097234103222
Hits @7: 0.4838038209295694
Hits left @8: 0.6095808383233533
Hits right @8: 0.38933561448531506
Hits @8: 0.4994582264043342
Hits left @9: 0.62178500142572
Hits right @9: 0.4045052751639578
Hits @9: 0.5131451382948389
Hits left @10: 0.6334759053321928
Hits right @10: 0.41790704305674364
Hits @10: 0.5256914741944682
Mean rank left: {0} 126.00530367835756
Mean rank right: {0} 251.68320501853435
Mean rank: {0} 188.84425434844596
Mean reciprocal rank left: {0} 0.45205307226264
Mean reciprocal rank right: {0} 0.24694124827874464
Mean reciprocal rank: {0} 0.3494971602706923
epoch : 454
epoch time: 58.4254
loss: 0.0013419411843642592
```

test:
```
--------------------------------------------------
test_evaluation
--------------------------------------------------

Hits left @1: 0.34813837584286134
Hits right @1: 0.15821362259356983
Hits @1: 0.2531759992182156
Hits left @2: 0.43516075442196817
Hits right @2: 0.2236880680152448
Hits @2: 0.3294244112186065
Hits left @3: 0.4856835727548129
Hits right @3: 0.2674679957001857
Hits @3: 0.37657578422749927
Hits left @4: 0.520521841102316
Hits right @4: 0.30171992573047984
Hits @4: 0.4111208834163979
Hits left @5: 0.5472002345353268
Hits right @5: 0.32913124206000194
Hits @5: 0.4381657382976644
Hits left @6: 0.5687481676927587
Hits right @6: 0.3510212059024724
Hits @6: 0.45988468679761557
Hits left @7: 0.5864360402618978
Hits right @7: 0.3685136323658751
Hits @7: 0.47747483631388643
Hits left @8: 0.6009479136128213
Hits right @8: 0.3865435356200528
Hits @8: 0.493745724616437
Hits left @9: 0.6140916642235903
Hits right @9: 0.40100654744454217
Hits @9: 0.5075491058340662
Hits left @10: 0.6266490765171504
Hits right @10: 0.41380826737027265
Hits @10: 0.5202286719437115
Mean rank left: {0} 137.1678393433011
Mean rank right: {0} 259.1085703117365
Mean rank: {0} 198.13820482751882
Mean reciprocal rank left: {0} 0.44235621231862876
Mean reciprocal rank right: {0} 0.24320714541485802
Mean reciprocal rank: {0} 0.34278167886674343
```