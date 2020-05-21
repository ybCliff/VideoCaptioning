import json

a = json.load(open("/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_NARFormer/EBN1_NDL1_WC20_MI_nv08_ei2_mp0.35_0.9/opt_info.json"))
b = json.load(open("/home/yangbang/VideoCaptioning/0219save/MSRVTT/IEL_NARFormer/EBN1_NDL1_WC20_MI_nv08_ei2_beta035_090_testing/opt_info.json"))

for k in b.keys():
    if k not in a.keys():
        print('key %s: %s' % (k, str(b[k])))
    else:
        if b[k] != a[k]:
            print('not equal %s %s %s', k, str(b[k]),str(a[k]))

print(b['nv_scale'])
print(a['nv_scale'])