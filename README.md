# transformer-MT
MT model implemented by pytorch

## environment

```
pytion=3.10
pytorch=2.0.0
pytorch-cuda=11.7
spacy=3.3.1
torchtext=0.15.0
```


## test data


|Source text|Targer text|
|------|---|
|[de_core_news_sm](https://spacy.io/models/de)|[en_core_web_sm](https://spacy.io/models/en)|


## result

Example 0 
```
Source Text (Input)        : `<s>` Ein Mann sitzt mit einem Bier in der Hand auf einem Stuhl und brät an einem `<unk>` etwas zu essen . `</s>`
Target Text (Ground Truth) : `<s>` A man sitting on a chair with a beer in his hands roasting something to eat on a wooden stick . `</s>`
Model Output               : `<s>` A man is sitting in a chair with a beer while eating something in an Asian food market . `</s>`
```

Example 1 
```
Source Text (Input)        : `<s>` Ein Kassierer steht an einem Schalter . `</s>`
Target Text (Ground Truth) : `<s>` A bank `<unk>` standing at a counter . `</s>`
Model Output               : `<s>` A crowded booth at a counter . `</s>`
```
Example 2 
```
Source Text (Input)        : `<s>` Zwei Frauen in Rot und ein Mann , der aus einer `<unk>` Toilette kommt . `</s>`
Target Text (Ground Truth) : `<s>` Two women wearing red and a man coming out of a port - a - `<unk>` . `</s>`
Model Output               : `<s>` Two women in red and a man passing a `<unk>` . `</s>`
```
Example 3 
```
Source Text (Input)        : `<s>` Eine Frau in einem schwarzen T-Shirt scheint eine Rolltreppe `<unk>` . `</s>`
Target Text (Ground Truth) : `<s>` A woman wearing a black t - shirt appears to being going up a escalator . `</s>`
Model Output               : `<s>` A woman in a black t - shirt appears to be an escalator . `</s>`
```
Example 4 
```
Source Text (Input)        : `<s>` Männer in blauer Spielkleidung sitzen in einem Bus . `</s>`
Target Text (Ground Truth) : `<s>` Men wearing blue uniforms sit on a bus . `</s>`
Model Output               : `<s>` Men in a blue uniform sitting on a bus . `</s>`
```