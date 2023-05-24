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


|Trial|Source text|Targer text|Language Pair|
|----|------|---|-----|
|1|[de_core_news_sm](https://spacy.io/models/de)|[en_core_web_sm](https://spacy.io/models/en)|DE => EN|
|2|[en_core_web_sm](https://spacy.io/models/en)|[de_core_news_sm](https://spacy.io/models/de)| EN => DE|


## result 1

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

## result 2
Example 0 ========
```
Source Text (Input)        : <s> A woman with black hair , wearing a black top and a red skirt is shaking her fist at somebody . </s>
Target Text (Ground Truth) : <s> Eine Frau mit schwarzem Haar , schwarzem Oberteil und einem roten Rock <unk> jemandem mit der Faust . </s>
Model Output               : <s> Eine Frau mit schwarzen Haaren , schwarzem Oberteil und einem roten Rock schüttelt sich die Schulter eines <unk> . </s>
```
```
Example 1 ========

Source Text (Input)        : <s> Two dogs run in a field looking at an unseen Frisbee . </s>
Target Text (Ground Truth) : <s> Zwei Hunde rennen über ein Feld und blicken dabei auf eine <unk> Frisbeescheibe . </s>
Model Output               : <s> Zwei Hunde laufen auf einem Feld und werden eine <unk> Frisbee - Frisbee zu . </s>
```
```
Example 2 ========

Source Text (Input)        : <s> A group of men , women , and children , all wearing hats , talk on the beach . </s>
Target Text (Ground Truth) : <s> Eine Gruppe von Männern , Frauen und Kindern , die alle Hüte tragen , unterhält sich am Strand . </s>
Model Output               : <s> Eine Gruppe von Männern , Frauen , Kinder und Kindern , die Hüte tragen , reden am Strand . </s>
```
```
Example 3 ========

Source Text (Input)        : <s> A man in an army uniform speaks into a microphone . </s>
Target Text (Ground Truth) : <s> Ein Mann in einer Armeeuniform spricht in ein Mikrofon . </s>
Model Output               : <s> Ein Mann in einem <unk> spricht in ein Mikrofon . </s>
```
```
Example 4 ========

Source Text (Input)        : <s> A man wearing headphones walks past a wall with red and purple graffiti . </s>
Target Text (Ground Truth) : <s> Ein Mann mit Kopfhörern geht an einer Mauer mit rotem und lilafarbenem Graffiti vorbei . </s>
Model Output               : <s> Ein Mann mit Kopfhörern geht an einer Wand mit roten und lila Graffiti vorbei . </s>
```

# ASR model
use well fine-tuned model
english speech to text

# Speech translation model
Load Speech Data(English) => ASR model => English text => MT model => Deutsch
2way
1. English speech(for ASR model) + pre-trained MT model
2. Speech dataset(for ASR model & MT model)