# Basic Problems in NLP (Notebook 1 & 2)

The best introduction to NLP or Natural Language Processing, is to understand the varieties of problems NLP wishes to solve, their categorical breakdowns, a summary of their previous approaches, and a brief overview of how these problems are tackled currently. In this series of notebooks, we hope to do the following things:

* Understand the fundamental problems in NLP
* Understand the divisions between the problems in NLP
* Understand the relationship between NLP and other fields of research
* Briefly introduce both statistical as well as linguistic approaches to solving a variety of the aforementioned problems
* And introduce the case study of Machine Translation and the scale of even a single problem in NLP


## The fundamental problems in NLP and their respective categories

Natural language processing is frequently subdivided into three main categories: Syntax, Semantics, and Pragmatics.

*Syntax* simply refers to the order or arrangement of words in a sentence such that they are grammatically correct. *Semantics* refers to the meaning that is conveyed by a piece of text.
*Pragmatics* is a subfield of linguistics that studies the ways in which context contributes to meaning. Unlike semantics, which examines meaning that is conventional or, in linguistic speak, "coded," in a given language, pragmatics studies how the transmission of meaning depends not only on linguistic as well as structural knowledge (vocabulary, grammar, etc.) of the speaker and the listener, but also on the context of the *utterance*, any pre-existing knowledge about those involved, the inferred intent of the speaker, and other factors. In this respect, pragmatics explains how language users are able to overcome *ambiguity*, since meaning relies on the how, where, when, etc. of an utterance.

The following is a list of some of the most commonly researched tasks in natural language processing. Some of these tasks have direct real-world applications, while others more commonly serve as subtasks that are used to aid in solving larger tasks.


### _*Syntax Tasks*_

Firstly, there are tasks related to syntax. These are the tasks that have to deal with, in essence, the order of words.

* *Lemmatization* - This is the task of removing “inflectional endings” (i.e. "-ed", "-s", "-ing") only and returning the base dictionary form of a word which, in linguistics, is also known as a lemma. This usually is a subtask, greatly aiding in other tasks like grammar induction. An example of this would be looking at the word "unwavering" and returning "waver" which is the base form of the word.

* *Morphological segmentation* - *Morphology* is a fancy way of saying the structure of words. Morphological segmentation basically means to separate words into their individual parts, or in linguistic speak, their individual *morphemes*, and identify the “class”, or type, of the morphemes. The difficulty of this task depends greatly on the complexity of the morphology of the language being considered. English has fairly simple morphology, especially inflectional morphology (meaning English doesn’t have many prefixes and suffixes that change the meaning of a word), and thus it is often possible to ignore this task entirely and simply model all possible forms of a word (e.g. "prickle, prickles, prickled, prickling") as separate words. In languages like Turkish and Arabic, however, such an approach is not possible, as each dictionary entry has thousands of possible word forms.

* *Part-of-speech (POS) tagging* - The task is simple and exactly what it sounds like: given a sentence, determine the part of speech (or POS, for short) for each word in that sentence. Obviously, in English, many words, especially common ones, can serve as multiple parts of speech. For example, "book" can be a noun ("the book on the table") or verb ("to book a hotel"); "bank" can be a noun ("river bank") or verb ("plane banking"); and "in" can be at least four different parts of speech. Some languages have more of this type of ambiguity than others. Languages with little inflectional morphology (look above for a quick definition), such as English, are particularly prone to such ambiguity. Chinese is prone to such ambiguity because it is a tonal language during verbalization. The tonal type of inflection is not easily conveyed through text, adding to the ambiguity of the sentence.

* *Grammar induction* - A task in which the end product is the generation of a formal grammar that describes a language's syntax; this task usually has multiple underlying pieces that fit together to create the “formal grammar.”

* *Parsing* - This task is relatively difficult and composed of multiple subtasks. The overview is to determine the parse tree or the "grammatical analysis" of a given sentence. This is similar to "diagramming" a sentence; understanding the words, parts of speech, how these fit together, what modifies what, etc. The grammar for natural languages is ambiguous and typical sentences have multiple possible analyses. In fact, perhaps surprisingly, for a typical sentence there may be thousands of potential parses (most of which will seem completely nonsensical to a human). There are two primary types of parsing, Dependency Parsing and Constituency Parsing. Dependency Parsing focuses on the relationships between words in a sentence (marking things like Primary Objects and predicates, whereas Constituency Parsing focuses on building out the Parse Tree using a Probabilistic Context-Free Grammar (PCFGs). 

* *Sentence breaking* - Given a chunk of text, find the sentence boundaries. Sentence boundaries are often marked by periods or other punctuation marks, but these same characters can serve other purposes (e.g. marking abbreviations).

* *Stemming* - The process of reducing inflected (or sometimes derived) words to their root form. (e.g. "close" will be the root for "closed", "closing", "close", "closer" etc).

* *Word segmentation* - Separate a chunk of continuous text into separate words. For a language like English, this is fairly trivial, since words are usually separated by spaces. However, some written languages like Chinese, Japanese and Thai do not mark word boundaries in such a fashion, and in those languages text segmentation is a significant task requiring knowledge of the vocabulary and morphology of the words in the language.

* *Terminology extraction* - The goal of terminology extraction is to automatically extract relevant terms from a given corpus.

### _*Semantic Tasks*_

* *Lexical semantics*
    * What is the computational meaning of individual words in context?
* *Distributional semantics*
    * How can we learn semantic representations from data?
* *Machine translation*
    * Automatically translate text from one human language to another. This is one of the most difficult problems, and is a member of a class of problems colloquially termed "AI-complete", i.e. requiring all of the different types of knowledge that humans possess (grammar, semantics, facts about the real world, etc.) in order to solve properly.
* *Named entity recognition (NER)*
    * Given a stream of text, determine which items in the text map to proper names, such as people or places, and what the type of each such name is (e.g. person, location, organization). Although capitalization can aid in recognizing named entities in languages such as English, this information cannot aid in determining the type of named entity, and in any case is often inaccurate or insufficient. For example, the first letter of a sentence is also capitalized, and named entities often span several words, only some of which are capitalized. Furthermore, many other languages in non-Western scripts (e.g. Chinese or Arabic) do not have any capitalization at all, and even languages with capitalization may not consistently use it to distinguish names. For example, German capitalizes all nouns, regardless of whether they are names, and French and Spanish do not capitalize names that serve as adjectives.
* *Natural language generation*
    * Convert information from computer databases or semantic intents into readable human language.
* *Natural language understanding*
    * Convert chunks of text into more formal representations such as first-order logic structures that are easier for computer programs to manipulate. Natural language understanding involves the identification of the intended semantic from the multiple possible semantics which can be derived from a natural language expression which usually takes the form of organized notations of natural language concepts. Introduction and creation of language metamodel and ontology are efficient however empirical solutions. An explicit formalization of natural language semantics without confusions with implicit assumptions such as closed-world assumption (CWA) vs. open-world assumption, or subjective Yes/No vs. objective True/False is expected for the construction of a basis of semantics formalization.
* *Question answering*
    * Given a human-language question, determine its answer. Typical questions have a specific right answer (such as "What is the capital of Canada?"), but sometimes open-ended questions are also considered (such as "What is the meaning of life?"). Recent works have looked at even more complex questions.
* *Recognizing Textual entailment*
    * Given two text fragments, determine if one being true entails the other, entails the other's negation, or allows the other to be either true or false.
* *Relationship extraction*
    * Given a chunk of text, identify the relationships among named entities (e.g. who is married to whom).
* *Sentiment analysis (see also multimodal sentiment analysis)*
    * Extract subjective information usually from a set of documents, often using online reviews to determine "polarity" about specific objects. It is especially useful for identifying trends of public opinion in the social media, for the purpose of marketing.
* *Topic segmentation and recognition*
    * Given a chunk of text, separate it into segments each of which is devoted to a topic, and identify the topic of the segment.
* *Word sense disambiguation*
    * Many words have more than one meaning; we have to select the meaning which makes the most sense in context. For this problem, we are typically given a list of words and associated word senses, e.g. from a dictionary or from an online resource such as WordNet.
* *Automatic summarization *
    * Produce a readable summary of a chunk of text. Often used to provide summaries of text of a known type, such as research papers, articles in the financial section of a newspaper.
* *Coreference resolution *
    * Given a sentence or larger chunk of text, determine which words ("mentions") refer to the same objects ("entities"). Anaphora resolution is a specific example of this task, and is specifically concerned with matching up pronouns with the nouns or names to which they refer. The more general task of coreference resolution also includes identifying so-called "bridging relationships" involving referring expressions. For example, in a sentence such as "He entered John's house through the front door", "the front door" is a referring expression and the bridging relationship to be identified is the fact that the door being referred to is the front door of John's house (rather than of some other structure that might also be referred to).
* *Discourse analysis*
    * This rubric includes a number of related tasks. One task is identifying the discourse structure of connected text, i.e. the nature of the discourse relationships between sentences (e.g. elaboration, explanation, contrast). Another possible task is recognizing and classifying the speech acts in a chunk of text (e.g. yes-no question, content question, statement, assertion, etc.).


### *Pragmatic Tasks*



## Ambiguity and compositionality

To properly understand what these mean to a practitioner of NLP, we have to fundamentally understand what the goal of NLP actually is and how we approach problems in NLP in general. These challenges require good design techniques; both modular approaches to break a problem up at appropriate points into smaller challenges, and the more formal models which reflect aspects of the structure of language. These problems are different and slightly more challenging because of two main aspects of language: ambiguity and compositionality.

Ambiguity can be referred as the ability of having more than one meaning or being understood in more than one way. Natural languages are ambiguous, so computers are not able to understand language the way people do. Natural Language Processing (NLP) is concerned with the development of computational models of aspects of human language processing. Ambiguity can occur at various levels of NLP. Ambiguity could be Lexical (word-level), Syntactic (dealing with order of words), Semantic (dealing with meaning of words), Pragmatic (dealing with contextual meanings) etc.

The sentence "You have the red light" is ambiguous. Without knowing the context, the identity of the speaker or the speaker's intent, it is difficult to infer the meaning with certainty. For example, it could mean:
                * the space that belongs to you has red ambient lighting;
                * you are stopping at a red traffic signal; and you have to wait to continue driving;
                * you are not permitted to proceed in a non-driving context;
                * your body is cast in a red glow; or
                * you possess a light bulb that is tinted red.

Similarly, the sentence "Sherlock saw the man with binoculars" could mean that Sherlock observed the man by using binoculars, or it could mean that Sherlock observed a man who was holding binoculars (syntactic ambiguity). The meaning of the sentence depends on an understanding of the context and the speaker's intent. As defined in linguistics, a sentence is an abstract entity—a string of words divorced from non-linguistic context—in constrast to an utterance, which is a concrete example of a speech act in a specific context. The more closely conscious subjects stick to common words, idioms, phrasings, and topics, the more easily others can surmise their meaning; the further they stray from common expressions and topics, the wider the variations in interpretations.

This suggests that sentences do not have intrinsic meaning, that there is no meaning associated with a sentence or word, and that either can only represent an idea symbolically. "The dog sat on the carpet" is a sentence in English. If someone were to say to someone else, "The dog sat on the carpet," the act is itself an utterance. This implies that a sentence, term, expression or word cannot symbolically represent a single true meaning; such meaning is underspecified (which dog sat on which carpet?) and potentially ambiguous. By contrast, the meaning of an utterance can be inferred through knowledge of its context, both its linguistic and non-linguistic contexts (which may or may not be sufficient to resolve ambiguity).

Compositionality is the other beast that presents itself as a basic roadblock in the field of NLP.

Computational semantics often seems like a field divided by methodologies and near-term goals. Logical approaches rely on techniques from proof theory and model-theoretic semantics, they have strong ties to linguistic semantics, and they are concerned primarily with inference, ambiguity, vagueness, and compositional interpretation of full syntactic parses. In contrast, statistical approaches derive their tools from algorithms and optimization, and they tend to focus on word meanings and broad notions of semantic content.

The two types of approaches share the long-term vision of achieving deep natural language understanding, but their day-to-day differences can make them seem unrelated and even incompatible. The distinction between logical and statistical approaches is rapidly disappearing, with the development of models that can learn the conventional aspects of natural language meaning from corpora and databases. These models interpret rich linguistic representations in a compositional fashion, and they offer novel perspectives on foundational issues like ambiguity, inference, and grounding. The fundamental question for these approaches is what kinds of data and models are needed for effective learning. Addressing this question is a prerequisite for implementing robust systems for natural language understanding, and the answers can inform psychological models of language acquisition and language processing. The leading player in the discussion is compositionality. After describing our view of linguistic objects (section 2), we introduce these two players (section 3). Although they come from different scientific worlds, we show that they are deeply united around the concepts of generalization, meaning, and structural complexity. The bulk of the paper is devoted to showing how learning-based theories of semantics bring the two worlds together. Specifically, compositionality characterizes the recursive nature of the linguistic ability required to generalize to a creative capacity, and learning details the conditions under which such an ability can be acquired from data. We substantiate this connection first for models in which the semantic representations are logical forms (section 4) and then for models in which the semantic representations are distributed (e.g., vectors; section 5). Historically, distributional approaches have been more closely associated with learning, but we show, building on much previous literature, that both types of representations can be learned. Our focus is on learning general theories of semantics, so we develop the ideas using formal tools that are familiar in linguistics, computer science, and engineering, and that are relatively straightforward to present in limited space: context-free grammars, simple logical representations, linear models, and firstorder optimization algorithms. This focus means that we largely neglect many important, relevant developments in semantic representation (de Marneffe et al. 2006; MacCartney & Manning 2009; van Eijck & Unger 2010; Palmer et al. 2010), semantic interpretation (Dagan et al. 2006; Saur´ı & Pustejovsky 2009), and structured prediction (Baklr et al. 2010; Smith 2011). It’s our hope, though, that our discussion suggests new perspectives on these efforts. (For more general introductions to data-driven approaches to computational semantics, see Ng & Zelle 1997; Jurafsky & Martin 2009: §IV.)

In linguistics, semantic representations are generally logical forms: expressions in a fully specified, unambiguous artificial language. The grammar in table 1 adopts such a view, defining semantic representations with a logical language that has constant symbols for numbers and relations and uses juxtaposition and bracketing to create complex expressions. In the literature, one encounters a variety of different formalisms — for example, lambda calculi (Carpenter 1997) or first-order fragments thereof (Bird et al. 2009), natural logics (MacCartney & Manning 2009; Moss 2009), diagrammatic languages (Kamp & Reyle 1993), programming languages (Blackburn & Bos 2005), robot controller languages (Matuszek et al. 2012b), and database query languages (Zelle & Mooney 1996). A given utterance might be consistent with multiple logical forms in our grammar, creating ambiguity. For instance, the utterance in line B of table 2 also maps to the logical form ¬(+ 3 1), which denotes −4. Intuitively, this happens if “minus” is parsed as taking scope over the addition expression to its right. Similarly, utterance C can be construed with “two times two” as a unit, leading to the logical form (− 2 (× 2 2)), which denotes −2. Utterance D also has an alternative analysis as (+ (+ 2 3) 4), but this ambiguity is spurious in the sense that it has the same denotation as the one in table 1. Our grammar also has one lexical ambiguity — “minus” can pick out a unary or binary relation — but this is immediately resolved in complex structures.

Thus far, we have allowed compositionality and learning to each tell its own story of generalization and productivity. We now show that the two are intimately related. Both concern the ability of a system (human or artificial) to generalize from a finite set of experiences to a creative capacity, and to come to grips with new inputs and experiences effectively. From this perspective, compositionality is a claim about the nature of this ability when it comes to linguistic interpretation, and learning theory offers a framework for characterizing the conditions under which a system can attain this ability in principle. Moreover, establishing the relationship between compositionality and learning provides a recipe for synthesis: the principle of compositionality guides researchers on specific model structures, and machine learning provides them with a set of methods for training such models in practice. More specifically, the claim of compositionality is that being a semantic interpreter for a language L amounts to mastering the syntax of L, the lexical meanings of L, and the modes of semantic combination for L. This also suggests the outlines of a learning task. The theories sketched above suggest a number of ways of refining this task in terms of the triples hu, s, di. We discuss two in detail. The pure semantic parsing task (section 4.1) is to learn an accurate mapping from utterances u to logical forms s. The interpretation task (section 4.2) is to learn an accurate mapping from utterances u to denotations d via latent semantic representations, in effect combining semantic parsing and interpretation. Throughout our review of the two tasks, we rely on the small illustrative example in figure 2. The figure is based around the utterance “two times two plus three”. Candidate semantic representations are given in row (a). The middle candidate y2 contains a lexical pairing that is illicit in the true, target grammar (table 1). This candidate is a glimpse into the unfettered space of logical forms that our learning algorithm needs to explore. Our feature vector counts lexical pairings and inspects the root-level operator, as summarized in the boxes immediately below each candidate. Row (b) of figure 2 describes how the semantic parsing model operates on these candidates, and row (c) does the same for the interpretation model. The next two subsections describe these processes in detail.


            * Context-free grammars
                * consists of non-terminal symbols (including a start symbol), terminal symbols, and productions, rewrite operation
                * derivation = sequence of rewrite operations
                * language defined by a CFG = sequences of terminal symbols derivable from start symbol
                * CFG as a device for
                    * generating sentences
                    * recognizing sentences
                    * parsing sentences
            * natural language grammars typically treat POS as terminal (or pre-terminal) and treat lexical insertion or look-up as a separate process
            * more powerful than regular expressions / finite-state automata
                * some languages which can be captured by CFG cannot be captured by regular expressions
                    * regular expressions can't capture center embedding
                * even if the language can be captured in principle by a reg. expr., it may not be convenient for expressing relations among constituents
        * data sparsity?


References:

* https://becominghuman.ai/a-simple-introduction-to-natural-language-processing-ea66a1747b32
* https://en.wikipedia.org/wiki/Pragmatics
* https://en.wikipedia.org/wiki/Syntax (https://en.wikipedia.org/wiki/Pragmatics)
* https://en.wikipedia.org/wiki/Semantics (https://en.wikipedia.org/wiki/Pragmatics)
* Artzi Y, Zettlemoyer LS. 2011. Bootstrapping semantic parsers from conversations. In Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing. Edinburgh: ACL
* Artzi Y, Zettlemoyer LS. 2013. Weakly supervised learning of semantic parsers for mapping instructions to actions. Transactions of the Association for Computational Linguistics 1:49–62
* Baklr G, Hofmann T, Sch¨olkopf B, Smola AJ, Taskar B, eds. 2010. Predicting Structured Data. Cambridge, MA: MIT Press
* Baroni M, Bernardi R, Do NQ, Shan Cc. 2012. Entailment above the word level in distributional semantics. In Proceedings of the 13th Conference of the European Chapter of the Association for Computational Linguistics. Avignon, France: ACL
* Berant J, Chou A, Frostig R, Liang P. 2013. Semantic parsing on Freebase from question–answer pairs. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing. Seattle: ACL
* Berant J, Liang P. 2014. Semantic parsing via paraphrasing. In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics Human Language Technologies. Baltimore: ACL
* Bird S, Klein E, Loper E. 2009. Natural Language Processing with Python. Sebastopol, CA: O’Reilly Media Blackburn P, Bos J. 2003. Computational semantics. Theoria 18:27–45
* Dowty D. 2007. Compositionality as an empirical problem. In Direct Compositionality, eds. C Barker, P Jacobson. Oxford: Oxford University Press, 23–101
* van Eijck J, Unger C. 2010. Computational Semantics with Functional Programming. Cambridge: Cambridge University Press
* https://web.stanford.edu/~cgpotts/manuscripts/liang-potts-semantics.pdf
* http://www.ijircce.com/upload/2014/sacaim/59_Paper%2027.pdf
