# 石原研究室用の流体解析のユーティリティ

このリポジトリは石原研での研究活動において解析したいデータをビッグデータから抽出するプログラムである。

## 出力形式

出力形式は `--format csv` または `--format vtk` で指定できる。省略時は出力ファイル名の拡張子から判定する。CSVでは取得した列をそのまま出力し、VTKでは従来どおりスカラー場・ベクトル場として出力する。`--velocity-gradient` の出力はCSVのみ対応する。

```sh
python main.py --dataset keta2_4096 --output boxcell.csv --format csv \
  1 1 1 64 gnx gny nn u v w
```

## boxcellの速度勾配

`main.py` の `--velocity-gradient` を指定すると、取得したboxcellの各軸の面中心6点を使い、boxcellにつき1組の速度勾配テンソルを計算してCSVに出力する。格子間隔は `coarsen_grid.py` と同じ角度座標の式 `2π × (size - 1) / data_size` を使用する。`size` は4以上の偶数を指定する。

```sh
python main.py --dataset keta2_4096 --output velocity_gradient.csv \
  --velocity-gradient 1 1 1 64 u v w
```

CSVにはboxcellの原点 (`gnx`, `gny`, `nn`)、速度勾配テンソル `∇u` の9成分 (`u_x_grad` から `w_z_grad`)、対称なひずみ速度テンソル `S = (∇u + ∇uᵀ) / 2` (`s_xx` から `s_zz`)、反対称な渦度テンソル `Ω = (∇u - ∇uᵀ) / 2` (`omega_xx` から `omega_zz`)、および `coarsen_grid.py` と同じ定義の渦度ベクトル (`vorticity_x`, `vorticity_y`, `vorticity_z`) が1行で出力される。
