### Character Level Language model
This project is based on the paper 'Attention is All you Need'. I have designed an decoder that captured Character-Level semantics. The decoder is trained on Shakespeare way of writing. Below is the results.
##### Before Training
#
```in
Sr?qP-QWktXoL&jLDJgOLVz'RIoDqHdhsV&vLLxatjscMpwLERSPyao.qfzs$Ys$zF-w,;eEkzxjgCKFChs!iWW.ObzDnxA Ms$3
```
##### After Training
#
```out
MAISTINIUS:
and as thrum know gutay e if and be row,
KI my Welpstit this ne owat sweth at peack yalll
LEbeflidg Fread!' gane, wift mad's bigh will seach.
Withers plat ond theee crusteerin'd in.
Coome Imen.
MICICINIUS:
Lird more not in the my wo!

DUTES:

WAMARDectior, eright: the more this cede ready themanes
shall Whan deate cot-ine toold tames

KING HENRY BOMPULO:

Tors herizeds ing thid he sore.
I bllow fowell coluch boy,
Alll stiungstce hen himbs sto alll will ha fateed.
Whee cemey mes deand in these why goors sawn havon
Whats word erear?

JOHNIN:
Nrot, teneamer;
Proocith!
```

The trained output text may not make any sense as the training is based on Character Level Tokens rather that word level tokens. Here, an interesting point to note is that, from the top it looks like English language (Shakespeare-like) in the sense of average length of words, starting letter is Capital, inside letters are small case and punctuation also mostly makes sense. Also, it can be seen that the text generates is like conversation between people.

##### RUN
Clone this repo and run the main.py. It will automatically run on GPU if its available. If you want to train the model, delete the checkpoint from the checkpoint folder and run the main.py file. By default the saved checkpoint will be used to produce similar results.

