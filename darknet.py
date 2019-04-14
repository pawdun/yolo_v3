import re
import json
import torch
import torch.nn as nn


class DarkNetParser:
    def __init__(self, cfg_path, json_storage=True):
        if json_storage:
            with open('cfg/conf.json', 'w')as f:
                f.write(json.dumps(DarkNetParser.parse_cfg(cfg_path)))

    @staticmethod
    def parse_cfg(cfg_path):
        conf = {"params": []}
        _file = open(cfg_path, 'r')
        params = _file.read().split('\n')
        params = [x.strip() for x in params if not x.startswith('#')]
        params = [x for x in params if x is not '']
        _key = ""
        idx = -1
        for line in params:
            if re.search("^\[.*\]$", line):
                idx += 1
                _key = line[1:-1]
                conf['params'].append({_key:dict()})
                ref = conf['params'][idx][_key]
                continue
            name, value = line.split("=")
            name = name.strip()
            value = value.strip()
            if name == 'anchors':
                value = value.split(" ")
                value = [val for val in value if val != '']
                ref[name] = []
                for val in value:
                    tmp = val.split(',')
                    ref[name].append((int(tmp[0]), int(tmp[1])))
            elif name in ["activation",  "policy"]:
                ref[name] = value
            else:
                value = value.split(',')
                if len(value) > 1:
                    ref[name] = [float('0.{}'.format(x[1:])) if x.startswith('.')
                                   else int(x) for x in value]
                else:
                    value = value[0]
                    if '.' in value:
                        if value.startswith('.'):
                            value = '0.{}'.format(value[1:])
                        ref[name] = float(value)
                    else:
                        ref[name] = int(value)
        return conf

    @staticmethod
    def create_modules(json_conf):
        output_filters = []
        blocks = json_conf['params']
        net_info = blocks[0]['net']
        module_list = nn.ModuleList()
        #RGB image - 3 channels
        prev_filters = 3
        for idx, x in enumerate(blocks[1:]):
            module = nn.Sequential()
            k, values = list(x.items())[0]
            print('idx={}, prev_filters={}, type={}'.format(idx, prev_filters, k))
            # CONVOLUTIONAL LAYER
            if k == 'convolutional':
                activation = values['activation']
                try:
                    batch_norm = values['batch_normalize']
                    bias = False
                except:
                    batch_norm = 0
                    bias = True
                out_filters = values['filters']
                kernel_size = values['size']
                padding = values['pad']
                stride = values['stride']

                if padding:
                    pad = (kernel_size - 1) // 2
                else:
                    pad = 0

                conv = nn.Conv2d(in_channels=prev_filters, out_channels=out_filters,
                                 kernel_size=kernel_size, stride=stride, padding=pad, bias=bias)
                module.add_module('conv_{}'.format(idx), conv)
                if batch_norm:
                    bn = nn.BatchNorm2d(out_filters)
                    module.add_module('batch_norm_{}'.format(idx), bn)
                if activation == "leaky":
                    actv = nn.LeakyReLU(0.1, inplace=True)
                    module.add_module('relu_{}'.format(idx), actv)
            # UPSAMPLE LAYER
            elif k == "upsample":
                stride = values["stride"]
                upsample = nn.Upsample(scale_factor=stride, mode="bilinear")
                module.add_module('upsample_{}'.format(idx), upsample)
            # ROUTE LAYER
            elif k == "route":
                route = EmptyLayer()
                module.add_module('route_{}'.format(idx), route)
                try:
                    out_filters = sum([output_filters[x] for x in values['layers']])
                except:
                    out_filters = output_filters[values['layers']]
            elif k == "shortcut":
                shortcut = EmptyLayer()
                module.add_module('shortcut_{}'.format(idx), shortcut)
                out_filters = output_filters[values['from']]

            elif k == "yolo_layer":
                detection_layer = YOLOLayer(anchors=values['anchors'],
                                            nb_classes=80,
                                            img_width=net_info['width'],
                                            img_height=net_info['height'],
                                            mask=values['mask'],
                                            ignore_tresh=values['ignore_tresh'])
                module.add_module('yolo_layer_{}'.format(idx),detection_layer)
            prev_filters=out_filters
            output_filters.append(out_filters)
            module_list.append(module)

        return net_info, module_list


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class YOLOLayer(nn.Module):
    def __init__(self, anchors, nb_clasess=None, img_height=None,
                 img_width=None, mask=None, ignore_tresh=0.5):
        super(YOLOLayer, self).__init__()
        if mask is None: self.anchors = anchors
        else: self.anchors = [anchors[idx] for idx in mask]
        self.nb_classes = nb_clasess
        self.image_height = img_height
        self.image_width = img_width
        self.ingore_tresh = ignore_tresh
        self.mask = mask # which nchors are used

class YOLOModel(nn.Module):
    def __init__(self, module_list, config_darknet,  hyper_params):
        super(YOLOModel, self).__init__()
        self.config = DarkNetParser.parse_cfg(config_darknet)['params'][1:]
        self.modules, self.hyper_params = DarkNetParser.\
            create_modules(self.config)

    def forward(self, x, train):
        is_train = train is not None
        for idx, mod, conf in enumerate(zip(self.modules, self.config)):


if __name__ == '__main__':
    model = DarkNetParser('cfg/yolo_v3.cfg')
    print(model.modules)
