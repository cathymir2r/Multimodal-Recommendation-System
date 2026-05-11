# 多模态推荐系统

一个面向电商商品场景的多模态推荐系统，结合文本、图像与用户序列行为进行建模，并提供实验对比、推荐服务接口和可视化前端。

## 项目特点

- 多模态建模：融合商品文本特征、图像特征和用户历史行为
- 实验对比完整：支持 baseline、主模型、消融实验和结果汇总
- 服务化部署：提供 Django API、Vue 前端和可选的 MySQL / Redis 基础设施
- 工程化脚本：包含后端初始化、实验报告生成、前后端启动脚本

## 技术栈

- 模型与实验：Python、PyTorch、NumPy、Pandas
- 后端：Django
- 前端：Vue 3、Vue Router、Vite
- 可选基础设施：MySQL、Redis、Docker Compose

## 目录结构

```text
.
├─backend/                  Django 后端
│  ├─apps/recommendation/   推荐业务、接口与管理命令
│  ├─config/                Django 配置
│  ├─manage.py
│  └─requirements.txt
├─frontend/                 Vue + Vite 前端
│  ├─src/
│  ├─package.json
│  └─vite.config.js
├─deploy/                   部署文件
│  └─docker-compose.yml
├─scripts/                  一键脚本
├─templates/                模板文件
├─main.py                   主训练入口
├─train_recommender.py      推荐模型训练
├─baseline_experiments.py   基线实验
├─ablation_experiments.py   消融实验
├─compare_experiments.py    对比实验
└─README_SYSTEM.md          详细运行说明
```

## 主要功能

### 1. 模型训练与实验

支持以下研究与实验流程：

- 文本特征建模
- 图像特征建模
- 序列行为建模
- 多模态注意力融合
- 对齐损失与融合策略对比
- baseline / 主模型 / 消融实验自动汇总

常用命令示例：

```powershell
conda run -n muti python main.py resnet50
conda run -n muti python main.py vit_b_16
conda run -n muti python baseline_experiments.py
conda run -n muti python train_recommender.py
conda run -n muti python ablation_experiments.py
conda run -n muti python compare_experiments.py
```

也可以使用脚本一键生成实验报告：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\generate_experiment_report.ps1 -RunBaselines -RunMainModel -RunAblation
```

### 2. 推荐服务接口

后端默认提供以下接口：

- `GET /api/dashboard/`
- `GET /api/status/`
- `GET /api/experiments/`
- `GET /api/catalog/`
- `GET /api/products/<item_id>/`
- `GET /api/users/<user_id>/`
- `GET /api/recommendations/history/`
- `POST /api/recommend/`

### 3. 可视化前端

前端包含以下主要页面或模块：

- 系统总览
- 实验结果展示
- 推荐工作台
- 商品详情
- 用户与历史行为查看
- 系统状态监控

## 环境准备

### 后端依赖

`backend/requirements.txt`：

- Django >= 5.2, < 5.3
- pandas >= 2.2
- numpy >= 1.26
- torch >= 2.0
- requests >= 2.31
- pymysql >= 1.1
- redis >= 5.0

安装示例：

```powershell
cd backend
pip install -r requirements.txt
```

### 前端依赖

```powershell
cd frontend
npm install
```

## 快速启动

### 1. 配置后端环境变量

将 [backend/.env.example](/d:/多模态推荐系统/backend/.env.example) 复制为 `backend/.env`，按需修改：

- `DB_ENGINE=sqlite` 使用本地 SQLite
- `DB_ENGINE=mysql` 使用 MySQL
- `ENABLE_REDIS_CACHE=true` 启用 Redis 缓存

### 2. 启动可选基础设施

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_infra.ps1 -Detach
```

或手动执行：

```powershell
cd deploy
docker compose up -d
```

### 3. 初始化后端数据

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_backend.ps1 -SyncLimit 1000
```

### 4. 启动后端

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_backend.ps1
```

默认地址：

- Django: `http://127.0.0.1:8000/`

### 5. 启动前端

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_frontend.ps1
```

默认地址：

- Vite: `http://127.0.0.1:5173/`

## 输出产物

实验相关产物默认输出到：

- `outputs/text_features`
- `outputs/image_features_resnet50`
- `outputs/image_features_vit_b_16`
- `outputs/models`
- `outputs/reports`
- `outputs/reports/figures`

## 说明

- 仓库中的大体量数据集、日志、数据库和中间产物不建议纳入版本控制
- 如果需要更细的运行步骤，请查看 [README_SYSTEM.md](/d:/多模态推荐系统/README_SYSTEM.md)

## License

如需开源发布，建议补充许可证文件，例如 `MIT` 或 `Apache-2.0`。
