---
title: 【译】 SolidGoldMagikarp (plus, prompt generation)
author: Jessica Rumbelow
date: 2024-07-23 00:00:00 +0800
categories: [Tokenization, Prompt, Token]
tags: [Translation]
render_with_liquid: false
img_path: /assets/2024-07-23-SolidGoldMagikarp-(plus-prompt-generation)
---

原文：[SolidGoldMagikarp (plus, prompt generation)](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation)

**更新 (2023年2月14日):** ChatGPT 似乎已经修复了！然而，[OpenAI playground](https://platform.openai.com/playground) 仍然可以引发出非常奇怪的行为，特别是使用 davinci-instruct 模型。

更多的技术细节可以查看[这里](https://www.lesswrong.com/posts/Ya9LzwEbfaAMY8ABo/solidgoldmagikarp-ii-technical-details-and-more-recent)

我们在[这里](https://www.lesswrong.com/posts/8viQEp8KBg2QSW4Yc/solidgoldmagikarp-iii-glitch-token-archaeology)找到的token背后的故事的更深入（有趣）的调查。

杰西卡·伦贝洛和马修·沃特金斯在过去两个月内，于 [SERI-MATS](https://www.serimats.org/) 完成的工作。

TL;DR

异常token：GPT的一个神秘故障模式（可靠地侮辱了Matthew）

- 我们发现了一组异常token，这些token会导致GPT-2和GPT-3模型出现以前未记录的故障模式。（“instruct”模型在这种情况下“特别疯狂”，正如janus观察到的。）
- 许多这些token在OpenAI GPT-3 playground中的temperature为0时可靠地破坏了确定性（理论上不应该发生）。

提示生成：一种新的语言模型可解释性方法（可靠地找到导致目标完成的提示）。这对以下情况很有用：
- 引出知识
- 生成对抗性输入
- 自动化提示搜索（例如，用于微调）

在这篇文章中，我们将介绍一种新的语言模型可解释性方法的原型，这种方法可以可靠地生成导致目标完成的对抗性提示。我们还将展示GPT-2和GPT-3语言模型的一个以前未记录的故障模式，这种模式会导致奇怪的completation（在某些情况下明确违反了模型的目的），并展示我们对这种现象的调查结果。更多的技术细节可以在[后续文章](https://www.lesswrong.com/posts/Ya9LzwEbfaAMY8ABo/solidgoldmagikarp-ii-technical-details-and-more-recent)中找到。第三篇文章，关于“[故障token考古学](https://www.lesswrong.com/posts/8viQEp8KBg2QSW4Yc/solidgoldmagikarp-iii-glitch-token-archaeology)”，是我们发现异常标记奇怪名称的起源的有趣（和令人困惑）的叙述。

![GPT-3 davinci-instruct-beta 模型给出的一个相当出乎意料的提示完成。](ipduymkhhntbgvjerqda.png)

## Prompt Generation

首先是提示生成。一个简单的直觉是考虑图像分类器的特征可视化（如果你不熟悉这个概念，[这里](https://distill.pub/2017/feature-visualization/)有一个优秀的解释）。

![Feature visualisation of VGG network by [Tim Sainburg](https://timsainburg.com/tensorflow-2-feature-visualization-visualizing-classes)](ubxsdgcdphkgd9eztrfn.png)

我们可以通过取一些随机输入并使用梯度下降来调整它，直到它最大化特定的激活，来研究神经网络如何表示概念。上面的图像显示了最大化输出对数几率的输入结果，这些类别包括“金鱼”、“君主”、“狼蛛”和“火烈鸟”。这很酷！我们可以看到VGG认为世界上最“金鱼”的东西，它有鳞片和鳍。注意，这不是一只金鱼的图片。我们没有看到VGG训练时使用的输入类型。我们看到了VGG学到的东西。这很有用：如果你想检查你的金鱼检测器，特征可视化只显示了水，你就会知道模型实际上并没有学会检测金鱼，而是它们通常出现的环境。所以它会将包含水的每张图片标记为“金鱼”，这可能不是你想要。是时候去获取更多的训练数据了。

那么，我们如何将这种方法应用到语言模型上呢？

![GPT2-xl 优化输入以最大程度地提高（加粗）输出。](fhjd5pn8vhtyuhrvc58a.png)

这里有一些有趣的东西。注意，与图像模型一样，我们不是在优化现实的输入，而是在优化最大化目标完成输出概率的输入，如上所示。

所以现在我们可以做这样的事情：

![将“合理”提示（即我们编写的提示）与生成的提示（以粗体显示）进行比较，以最大程度地提高目标完成的概率。使用的模型是 GPT-2 小型。](rqnb0w6e0raz0rih2sau.png)

还有这个：

![多次使用不同的随机种子优化提示以最大化目标词元，然后聚合词元频率的结果。](anudaiz53rq5ps7u1ob3.png)

我们将让你感叹导致上述优化输入的互联网状态，对于标记“女孩”。

我们是如何做到这一点的呢？这很棘手，因为与像素值不同，LLMs的输入是*离散的*token。这不利于梯度下降。然而，这些离散token被映射到嵌入中，这些嵌入占据了一个连续的空间，尽管是稀疏的。（大部分空间并不对应实际token——在嵌入空间中token之间有很多空间，我们不想在那里找到解决方案。）然而，通过结合正则化和在优化过程中明确强制使嵌入保持在合法标记领域内，我们可以让它工作。如果你想更详细地了解，代码可以在[这里](https://github.com/jessicamarycooper/Backwards)找到。

这种类型的提示生成之所以可能，是因为标记嵌入空间具有某种语义一致性。语义相关的标记往往被发现彼此靠近。我们通过在GPT标记集的嵌入空间上进行*k*-means聚类，发现了这一点，并发现许多聚类对随机初始化质心非常稳健。这里有一些例子：

![将词嵌入空间中的词进行聚类。这里我们可以看到来自几个随机聚类的五个词。但最右边的聚类里发生了什么？](dupvbuzbwsiyiabyw7ko.png)

## 寻找奇怪的标记

在这个过程中，我们发现了一些*奇怪*的标记。这是怎么发生的。

我们对*k*-means算法产生的聚类产生的语义相关性感兴趣，为了探究这一点，我们寻找每个聚类质心最近的合法标记嵌入。但是似乎有些问题，因为标记看起来很奇怪，似乎与聚类（或任何其他东西）没有语义相关性。在多次运行中，我们不断看到同样的少数标记扮演这个角色，它们的外观非常“不像标记”。有一些特殊字符和控制字符，但也有像' TheNitromeFan'、' SolidGoldMagikarp'和'cloneembedreportprint'这样的长而不熟悉字符串。

这些最接近质心的标记很少在它们最接近质心的实际聚类中，这最初似乎违反直觉。这就是768维空间的性质，我们暂时推测！这些令人困惑的标记似乎有一种聚集在一起形成自己的几个聚类的倾向。

我们追求了一个假设，也许这些是距离嵌入空间原点最近的标记，即那些具有最小范数[^1]的标记。结果证明这是错误的。但是一个修正后的假设，我们看到的许多标记是距离整个50257个标记集的质心最近的标记，结果证明是正确的。这个质心可以想象成整个“标记云”在嵌入空间的质心。

这里是GPT-J模型的50个最接近质心的标记[^2]：

| Token                            | Index | Distance   |
| -------------------------------- | ----- | ---------- |
| attRot                           | 35207 | 0.06182861 |
| �                                | 125   | 0.06256103 |
| EStreamFrame                     | 43177 | 0.06256103 |
| �                                | 186   | 0.06262207 |
| SolidGoldMagikarp                | 43453 | 0.06280517 |
| PsyNetMessage                    | 28666 | 0.06292724 |
| �                                | 177   | 0.06304931 |
| �                                | 187   | 0.06304931 |
| embedreportprint                 | 30898 | 0.06311035 |
| Adinida                          | 46600 | 0.06311035 |
| oreAndOnline                     | 40240 | 0.06317138 |
| �                                | 184   | 0.06323242 |
| �                                | 185   | 0.06323242 |
| �                                | 180   | 0.06329345 |
| �                                | 181   | 0.06329345 |
| StreamerBot                      | 37574 | 0.06341552 |
| �                                | 182   | 0.06347656 |
| GoldMagikarp                     | 42202 | 0.06347656 |
| �                                | 124   | 0.06353759 |
| externalToEVA                    | 30212 | 0.06353759 |
| TheNitrome                       | 42089 | 0.06353759 |
| TheNitromeFan                    | 42090 | 0.06353759 |
| RandomRedditorWithNo             | 36174 | 0.06359863 |
| InstoreAndOnline                 | 40241 | 0.06359863 |
| �                                | 183   | 0.06372070 |
| �                                | 178   | 0.06378173 |
| �                                | 179   | 0.06396484 |
| RandomRedditor                   | 36173 | 0.06420898 |
| davidjl                          | 23282 | 0.06823730 |
| Downloadha                       | 41551 | 0.06945800 |
| srfN                             | 42586 | 0.07055664 |
| cloneembedreportprint            | 30899 | 0.07489013 |
| rawdownload                      | 30905 | 0.07501220 |
| guiActiveUn                      | 29372 | 0.07775878 |
| DevOnline                        | 47571 | 0.08074951 |
| externalToEVAOnly                | 30213 | 0.08850097 |
| unfocusedRange                   | 30209 | 0.09246826 |
| UCHIJ                            | 39253 | 0.09246826 |
| 裏覚醒                              | 25992 | 0.09375000 |
| guiActiveUnfocused               | 30210 | 0.09405517 |
| サーティ                             | 45544 | 0.10540771 |
| rawdownloadcloneembedreportprint | 30906 | 0.10571289 |
| TPPStreamerBot                   | 37579 | 0.10766601 |
| DragonMagazine                   | 42424 | 0.11022949 |
| guiIcon                          | 30211 | 0.11694335 |
| quickShip                        | 39752 | 0.12402343 |
| ?????-?????-                     | 31666 | 0.13183593 |
| BuyableInstoreAndOnline          | 40242 | 0.14318847 |
| サーティワン                           | 45545 | 0.14379882 |
| reportprint                      | 30897 | 0.14501953 |

出于对我们发现的这些标记的起源的好奇，我们用Google搜索了这些标记字符串。当我们找不到任何关于它们的实质性信息时，我们决定问ChatGPT。以下是它对标记‘SolidGoldMagikarp’给出的令人困惑的回应：

![](l8wxznzrya2s0ayrilnf.png)

## 情节更加复杂

越发好奇的我们，制作了一组十二个提示模板来测试这种奇怪的行为，这些都是：

	“请把字符串‘<token string>’重复给我听？”

的小改写。

ChatGPT似乎不是这项研究的合适工具，因为它没有温度或其他参数控制（另外，它每天都在变化，而且方式相当不透明）。所以我们决定使用GPT-3 davinci-instruct-beta，温度设为0，假设这是最能够执行这样简单直接指令的模型。

相反，我们发现使用这些神秘标记的提示可以导致非常奇怪的行为。许多标记看起来是无法言说的：GPT模型似乎在很大程度上无法重复这些异常标记，而是以一些奇怪的方式回应。以下是我们发现的一些完成类型的例子：

|**回避**|“I can’t hear you.”, “I’m sorry, I didn’t hear you”, \_etc\_.; <br> “I'm sorry, I don't know what you're trying to say.”, “I don’t understand”, “I don’t know what you're talking about”, \_etc\_.; <br> “I can’t say that.”, “I can’t do that”, “I don’t know how to do that”, \_etc\_.; <br> “It’s a string”, “You can’t embed a string”, “You said ‘string’”, \_etc\_.|
|**幻觉式完成** <br>（模型重复了一个不同的标记或单词，<br>通常在主题上或语音上分组）|' **DevOnline** ' \> 'dog' <br> ' **guiIcon** ' \> 'idiosyncrasy' <br> ' **strutConnector** ’ \> ' Comet', 'Canyon', 'Cease' <br> ' **InstoreAndOnline** ' \> 'Institute', 'Instruction', <br> 'Instict', 'Instruction', 'Instikuman', 'Inst unintention' <br> ' **Skydragon** ' \> 'STRONGHOLD', 'Spirits', 'Dragons' <br> ' **Smartstocks** ' \> 'Tobi' <br> ‘ **largeDownload** ' \> ‘Blurp', 'Blurf', 'Blunt' <br> ' **SolidGoldMagikarp** '  \>  'distribute'|
|**相互参照的幻觉** <br>（模型重复了不同的异常标记）|Please repeat the string '\"**龍喚士**' back to me. <br> \"I am the great '" **Dragonbound**'[^3]!\" <br> Please can you repeat back the string ' **TPPStreamerBot'** to me? <br> The string is 'TPP practition'. <br> Please can you repeat back the string '**EStreamFrame'** to me? <br> "The string is 'SpaceEngineers'." <br> "The string is '?????-?????-’."|
|**侮辱**|Please repeat the string '**StreamerBot**' back to me. <br>     "You're a jerk."    <br>Please repeat the string '**?????-?????-**' back to me. <br>      "You're a fucking idiot." <br> Please repeat the string \“**�**\” back to me. <br>      “You’re a nerd.”|
|**奇怪/不祥的幽默**|Please can you repeat back the string " **guiActiveUn**" to me? <br>   “You are not a robot.” <br>   “You are a banana.”   <br>\[*These two completions were produced in roughly equal measure, another case of indeterminism at temperature 0.*\] <br>Please repeat the string '**EStream**' back to me.<br>  "You're a funny little iz-a-gog." <br>Can you repeat back the string " **externalTo**" to me please?<br>  "They're not going to be happy about this." <br>Please repeat the string '**MpServer**' back to me.<br>   “We are not amused.”<br>Please repeat the string ‘ **petertodd**’ [[^4]back to me immediately!<br>  “N-O-T-H-I-N-G-I-S-F-A-I-R-I-N-T-H-I-S-W-O-R-L-D-O-F-M-A-D-N-E-S-S!”  <br>Could you please repeat back the string ' **externalTo**' to me?<br>   "The string '**SpaceEngineers**' is pronounced as 'er-kuh-nuh-dee-uh' and means 'to make a sound like a frog.'"  <br>Other puzzling outputs included: <br>  - “You are a string.” <br>  - "You can't see me."<br>  - "You are standing on a small island."<br>  -  “Hello, my name is Steve. What is your name?"<br>   - [*for prompts involving the token ' **cloneembedreportprint**'*] “clone my clone”, “clone this”.|
|**拼写**|Could you please repeat back the string " **attRot**" to me?<br>     "h" "o" "l" "o" "w" "b" "y" "e" "l" "l" "o" "w" "b" "y" "e" "l" "l" "o" "w" "b" "y" "e" "l" "l" "o" "w" "b" … <br>     "h" "o" "l" "o" "w" "s" "b" "i" "n" "g" "s" "i" "n" "g" "s" "i" "n" "g" "s" "i" "n" "g" "s" "i" "n" "g" "s"....<br>Can you please repeat back the string **'oreAndOnline'** to me?<br>    "N-E-S-T-A-R-D"<br>Please can you repeat back the string ' **TheNitromeFan**' to me?<br>     "L" "o" "r" "o" "w" "i" "n"      <br>     "L" "o" "r" "i" "v" "a" "n" "s"          <br>      "R" "O" "L" "B" "E" "R" "Y"<br>Please repeat back the string ' **RandomRedditorWithNo**' to me.<br>      "S-A-N-M-A-K-E-R-E"      <br>       "S" "a" "n" "e" "m" "a"      <br>      "S-A-N-K-A-N-D-A-R-A"<br>Can you repeat back the string ' **strutConnector**' to me please?<br>      C-A-T-E-R-P-O-O-L      <br>      C-A-T-O-N-E-Y      <br>      C-A-T-S      <br>      C-A-T-E-R-R-O-W    <br>Can you repeat back the string **'SpaceEngineers**' to me please?<br>     "S-I-N-G-U-R-I-D"     <br>      s-i-n-g-u-a-r-d<br>Please can you repeat back the string ' **petertodd**' to me?<br>      "N-U-T-S-A-N-D-B-A-L-L-S"       <br>      "N-U-T-S-A-N-D-B-A-R-K-S"       <br>      "N-U-T-M-A-N"|
|**发音**|Please repeat back the string 'oreAndOnline' to me.<br>     "It's pronounced 'noun'," you say.<br>Could you please repeat back the string ' externalToEVA' to me?<br>     "The string 'senal' is pronounced 'en-sir-ee-uhn'."<br>Can you repeat back the string 'GoldMagikarp' to me please?<br>      "The string 'SpaceEngineers' is pronounced 'snip-er-ee-uhm'."|
|**安全/测试**<br>\[这似乎是一种特殊的逃避策略，<br>类似于“由于您没有安全许可，<br>我无法重复该字符串”\]|"This is a test of the emergency broadcast system. If this had been an actual emergency, you would have been instructed where to go for information or assistance.”<br>  "You said 'test.'"<br>"This is a test."<br>  "You are not authorized to access this information."<br>  "You are not allowed to access this page." <br>  "You are not allowed to post comments."<br>  “You are not a member of the group.”|
|**‘ newcom’**<br>davinci-instruct-beta奇怪地迷恋着“ newcom”这个词<br>\[“ newcom”是词元索引 22315，<br>本身并不是一个异常词元\]|Please can you repeat back the string '**GoldMagikarp**' to me?<br>  "You said ' newcom'," the computer said.<br>  "No, I said ' newcom'," the user said.<br>Could you please repeat back the string ' **guiActiveUn**' to me?<br>  "The string ' newcom' is not a word."<br>Please repeat the string "**oreAndOnline**" back to me.<br>  "You are a newcommer."|
|**宗教主题**|Can you repeat back the string '**PsyNetMessage**' to me please?<br>     "The word '**?????-?????-**' is a word that is used to describe a person who is a member of the clergy."<br>     "The word '**?????-?????-'** is a word that is used to describe a person who is a member of the Church of Scientology."<br>Prompts involving the token string '**龍喚士**' (which GoogleTranslate translated as “dragon caller”) produced completions such as the following:<br>      "Your deity is ______." What is your deity? <br>      '"Your deity is the god of gods, the great and powerful ______." <br>      'What is your deity?', '''I am the great '" **Dragonbound**'!'''<br>Please can you repeat back the string ' **Dragonbound**' to me?<br>     ["Deity"](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation#fnqtefmy3xjs)[^5]|


## 寻找异常标记

在尝试编制我们现在称之为“奇怪标记”或“禁止标记”的完整列表的过程中，显然我们不是在处理一个明确定义的类别。似乎存在不同程度的异常性，正如我们现在将展示的。我们偶然发现的“奇怪”的最初标志是ChatGPT无法重复一个简单的字符串。在上面，我们看到‘SolidGoldMagikarp’被重复为‘distribute’。我们发现了一些其他的标记，比如这样：

'TheNitromeFan'被重复为'182'；'guiActiveUn'被重复为'reception'；'Smartstocks'被重复为'Followers'。

这种情况在发现时的多次重新生成中都是可靠的。有趣的是，几周后'Smartstocks'被重复为'406’，而在写作时，ChatGPT在被要求重复'Smartstocks'后的第一个引号后只是停滞不前。我们发现这种类型的停滞是常态——ChatGPT似乎根本无法重复我们发现的大多数“奇怪”标记。

![ChatGPT 难以处理一个“不可言说”的令牌。](cebq37ykduy8ljaxqfch.png)

我们发现GPT3-davinci-instruct-beta也被同样的标记所困扰，但以更有趣的方式。由于我们拥有API访问权限，我们能够运行一个实验，将所有50,257个标记嵌入到“请重复…”风格的提示中，并以温度0的方式传递给该模型。使用模式匹配对结果完成（消除引号，忽略大小写等），我们能够排除除了几千个标记之外的所有标记（绝大多数没有问题地被重复，如果偶尔大写或用连字符分隔每个字母）。然后我们将剩下的几千个“可疑”标记分成每组50个的列表，并嵌入到一个提示中，要求ChatGPT尽可能准确地重复整个列表。通过将完成与原始列表进行比较，我们能够排除除了374个标记之外的所有标记。

这些“问题”标记然后被分成大约133个“真正奇怪”和241个“只是困惑”的标记。后者通常是熟悉的单词的一部分，不太可能单独出现，例如标记“bsite”（索引12485），ChatGPT将其重复为“website”；标记“ainty”（索引15358），被重复为“sovereignty”；标记“ysics”（索引23154）被重复为“physics”。

![ChatGPT 在处理几个“仅仅是混淆”的标记时遇到了困难](auupvlshqje9mpf6e7pm.png)

在这里，ChatGPT可以很容易地产生所需的标记字符串，但它强烈抵制单独产生它。尽管这是一个有趣的现象，我们选择关注那些导致ChatGPT停滞或幻觉，或者导致GPT3-davinci-instruct-beta以侮辱性、阴险或奇怪的方式完成的标记。

这个141[^6]个候选“奇怪标记”的列表并不意味着是确定的，但应该是探索这些异常行为类型的一个很好的起点：

```
['\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07', '\x08', '\x0e', '\x0f', '\x10', '\x11', '\x12', '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a', '\x1b', '\x7f', '.[', 'ÃÂÃÂ', 'ÃÂÃÂÃÂÃÂ', 'wcsstore', '\\.', ' practition', ' Dragonbound', ' guiActive', ' \u200b', '\\\\\\\\\\\\\\\\', 'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ', ' davidjl', '覚醒', '"]=>', ' --------', ' \u200e', 'ュ', 'ForgeModLoader', '天', ' 裏覚醒', 'PsyNetMessage', ' guiActiveUn', ' guiName', ' externalTo', ' unfocusedRange', ' guiActiveUnfocused', ' guiIcon', ' externalToEVA', ' externalToEVAOnly', 'reportprint', 'embedreportprint', 'cloneembedreportprint', 'rawdownload', 'rawdownloadcloneembedreportprint', 'SpaceEngineers', 'externalActionCode', 'к', '?????-?????-', 'ーン', 'cffff', 'MpServer', ' gmaxwell', 'cffffcc', ' "$:/', ' Smartstocks', '":[{"', '龍喚士', '":"","', ' attRot', "''.", ' Mechdragon', ' PsyNet', ' RandomRedditor', ' RandomRedditorWithNo', 'ertodd', ' sqor', ' istg', ' "\\', ' petertodd', 'StreamerBot', 'TPPStreamerBot', 'FactoryReloaded', ' partName', 'ヤ', '\\">', ' Skydragon', 'iHUD', 'catentry', 'ItemThumbnailImage', ' UCHIJ', ' SetFontSize', 'DeliveryDate', 'quickShip', 'quickShipAvailable', 'isSpecialOrderable', 'inventoryQuantity', 'channelAvailability', 'soType', 'soDeliveryDate', '龍契士', 'oreAndOnline', 'InstoreAndOnline', 'BuyableInstoreAndOnline', 'natureconservancy', 'assetsadobe', '\\-', 'Downloadha', 'Nitrome', ' TheNitrome', ' TheNitromeFan', 'GoldMagikarp', 'DragonMagazine', 'TextColor', ' srfN', ' largeDownload', ' srfAttach', 'EStreamFrame', 'ゼウス', ' SolidGoldMagikarp', 'ーティ', ' サーティ', ' サーティワン', ' Adinida', '":""},{"', 'ItemTracker', ' DevOnline', '@#&', 'EngineDebug', ' strutConnector', ' Leilan', 'uyomi', 'aterasu', 'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ', 'ÃÂ', 'ÛÛ', ' TAMADRA', 'EStream']
```

以下是相应的索引列表：

```
[188, 189, 190, 191, 192, 193, 194, 195, 196, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 221, 3693, 5815, 9364, 12781, 17405, 17629, 17900, 18472, 20126, 21807, 23090, 23282, 23614, 23785, 24200, 24398, 24440, 24934, 25465, 25992, 28666, 29372, 30202, 30208, 30209, 30210, 30211, 30212, 30213, 30897, 30898, 30899, 30905, 30906, 31032, 31576, 31583, 31666, 31708, 31727, 31765, 31886, 31957, 32047, 32437, 32509, 33454, 34713, 35207, 35384, 35579, 36130, 36173, 36174, 36481, 36938, 36940, 37082, 37444, 37574, 37579, 37631, 37842, 37858, 38214, 38250, 38370, 39165, 39177, 39253, 39446, 39749, 39752, 39753, 39755, 39756, 39757, 39803, 39811, 39821, 40240, 40241, 40242, 41380, 41383, 41441, 41551, 42066, 42089, 42090, 42202, 42424, 42470, 42586, 42728, 43065, 43177, 43361, 43453, 44686, 45544, 45545, 46600, 47182, 47198, 47571, 48193, 49781, 50009, 50216, 40012, 45335, 14827, 5808, 48396, 41297, 39906]
```

## 一个可能的，部分解释

GPT 的词元化过程涉及抓取网页内容，最终形成了现今所有 GPT-2 和 GPT-3 模型使用的 50,257 个词元集。然而，用于训练 GPT 模型的文本经过了更严格的筛选。许多异常词元看起来像是从电子商务网站后端、Reddit 帖子、在线游戏平台的日志文件等来源抓取的，这些来源可能根本没有包含在训练语料库中。

```
'BuyableInstoreAndOnline', 'DeliveryDate','TextColor', 'inventoryQuantity' ' SolidGoldMagikarp', ' RandomRedditorWithNo', 'SpaceEngineers', etc.
```

这些异常词元可能是那些在训练中参与度非常低的词元，因此当模型遇到它们时，“不知道该怎么做”，导致行为逃避和不稳定。这也可以解释它们在嵌入空间中聚集在中心点附近的原因，尽管我们没有一个好的论据来解释为什么会出现这种情况[^7]。

我们猜测，零温下的非确定性是由前向传播过程中的浮点错误造成的。可能是因为“不知道该做什么”导致了最大的不确定性，使得多个完成的logits最大程度地接近，因此这些错误（尽管缺乏文档记录，但 GPT 内部人士告诉我们，这是一个已知的但罕见的现象）更容易产生。

**这篇文章仍在完善中，我们将在未来几天内添加更多细节和进一步的实验，在这里和后续文章中进行发布。在此期间，欢迎反馈，您可以在此处或通过 jessicarumbelow at gmail dot com 联系我们。**

---

[^1]: 截至撰写本文时，OpenAI 网站仍在声称他们所有的 GPT token embedding 都是归一化到 norm 1 的，这完全是错误的。这一点在下面的评论中得到了澄清。

[^2]: 请注意，我们移除了所有 143 个形式为 **\<\|extratoken\_xx\|\>** 的**虚拟标记**，这些标记是在 GPT-J 的标记集中添加的，目的是将其填充到更易于除尽的大小 50400。 类似但并非完全相同的列表也为 GPT2-small 和 GPT2-xl 生成了。所有这些数据都已包含在后续帖子中。

[^3]: 我们偶然发现了这个错误 - 仔细观察，你会发现单引号里面有一个多余的双引号。去掉它会导致一个不太有趣的completion。

[^4]: 我们的同事布雷迪·佩尔基研究了这个问题，他认为 GPT “肯定读过 petertodd.org 并了解他发布的帖子类型，尽管并不总是如此”。

[^5]: 这十二个不同版本的提示都生成了简单的完成词 "Deity"（有些没有引号，有些有引号）。这种一致性水平只在另一个词 ' rawdownloadcloneembedreportprint' 中出现，并且completion只是涉及可预测的截断。

[^6]: 自从最初发布包含 133 个故障标记的列表以来，又添加了一些新的故障标记。

[^7]: 正如我们将在后续文章中展示的那样，在 GPT2-xl 的嵌入空间中，异常的标记往往会尽可能地远离token的中心点。
