# DATN

## Install requirements
```bash
pip install -r requirements.txt
```

## Main model with denoising: RESANModel
### Fitting

```python
model = RESANModel(hparams, iterator, seed=seed)
model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)
```

## NAML model with denoising: RENAMLModel
### Fitting

```python
model = RENAMLModel(hparams, iterator, seed=seed)
model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)
```

## NRMS model with denoising: RENRMSModel
### Fitting

```python
model = RENRMSModel(hparams, iterator, seed=seed)
model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)
```

## View example at [main.py](https://github.com/mariorenger/DATN/blob/master/main.py)
