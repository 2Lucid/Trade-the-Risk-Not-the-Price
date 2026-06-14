# LinkedIn pack — *Trade the Risk, Not the Price*

Everything you need to publish. Quick checklist:

1. **New post → Document** and upload **`linkedin/carousel.pdf`** (7 slides). Title it
   *"Trade the Risk, Not the Price"*.
2. Paste the **post text** below (pick FR or EN).
3. Put the **links in the FIRST COMMENT** (LinkedIn throttles posts with links in the
   body) — text is ready under "First comment".
4. Add the **hashtags** (already at the end of the post).
5. Optional: post the **follow-up (Post 2)** a few days later to build a mini-series.

> Replace `YOUR-SITE-URL` with your Vercel URL once deployed.

---

## 🇫🇷 Post principal (français)

> Tu ne peux pas prédire la bourse. Alors j'ai arrêté d'essayer — et le résultat est meilleur.

Pendant quelques semaines, j'ai reconstruit de zéro une vraie stratégie publiée dans le *Journal of Finance* (Moreira & Muir, 2017) et je l'ai stress-testée comme le ferait une table quant. Un seul chiffre a tout recadré pour moi :

→ Prédire le **rendement** du mois prochain, en out-of-sample : R² ≈ –12 % (pire qu'un pile ou face).
→ Prédire la **volatilité** du mois prochain : R² ≈ +28 %.

Tu ne peux pas prédire le prix. Mais tu peux prédire le risque. Et ça suffit.

L'idée est presque gênante de simplicité : investir **plus** quand le marché est calme, **moins** quand il est agité — sans jamais parier sur la direction. Comme ralentir sur une route verglacée.

Les résultats, à risque égal avec le buy-and-hold :
• Marché US : Sharpe 0,48 → 0,61, drawdown max réduit de –54 % à –43 %
• Sur 11 actifs mondiaux + une couche de trend-following, en portefeuille diversifié : Sharpe 0,39 → 0,72, alpha de 8 %/an (t = 3,6), drawdown quasi divisé par deux — et encore +0,64 net de coûts de transaction.

Ce dont je suis le plus fier, c'est la rigueur, pas les rendements :
• Strictement out-of-sample, zéro lookahead
• Coûts de transaction et plafonds de levier réalistes
• Deflated Sharpe Ratio (corrige le « j'ai testé plein de modèles »)
• Deux revues de code adversariales par IA pour chasser les bugs

Le bonus qui m'a sincèrement surpris : j'ai donné à un algorithme (un Hidden Markov Model) uniquement les rendements — aucune date de crise — et il a redécouvert tout seul la bulle internet, 2008, le COVID et 2022.

Je suis aussi honnête sur les limites : le vol-timing ne marche pas partout, l'edge est modeste seul, et un backtest n'est pas du trading réel. C'est de la recherche, pas un conseil d'investissement.

J'ai construit un **site interactif** pour que tu puisses jouer avec toi-même : choisis l'actif, le levier, les coûts, et regarde la stratégie se recalculer en direct.

🔗 Site + papier + code en premier commentaire.

Fais défiler le carrousel pour les 5 graphiques qui racontent l'histoire 👇

#Quant #FinanceQuantitative #DataScience #Python #Investissement #MachineLearning #Finance

---

## 🇬🇧 Main post (English)

> You can't predict the stock market. So I stopped trying — and got a better result.

For a few weeks I rebuilt a real *Journal of Finance* strategy from scratch (Moreira & Muir, 2017) and stress-tested it the way a quant desk would. One number reframed everything for me:

→ Forecasting next month's **return**, out-of-sample: R² ≈ –12% (worse than a coin flip).
→ Forecasting next month's **volatility**: R² ≈ +28%.

You can't predict the price. But you can predict the risk. And that's enough.

The idea is almost embarrassingly simple: invest **more** when markets are calm, **less** when they're turbulent — never guessing direction. Like slowing down on an icy road.

The results, at the same risk as buy-and-hold:
• US market: Sharpe 0.48 → 0.61, max drawdown cut from –54% to –43%
• Across 11 global assets + a trend overlay, diversified: Sharpe 0.39 → 0.72, alpha of 8%/yr (t = 3.6), drawdown nearly halved — still +0.64 net of trading costs.

What I'm most proud of is the rigor, not the returns:
• Strictly out-of-sample, zero lookahead
• Realistic transaction costs and leverage caps
• Deflated Sharpe Ratio (corrects for "I tried many models")
• Two adversarial AI code reviews to hunt for bugs

The bonus that genuinely surprised me: I fed an algorithm (a Hidden Markov Model) nothing but returns — no crisis dates — and it rediscovered the dot-com bust, 2008, COVID and 2022 on its own.

I'm honest about the limits too: vol-timing doesn't work everywhere, the edge is modest on its own, and a backtest is not live trading. This is research, not advice.

I built an **interactive site** so you can play with it yourself — pick the asset, the leverage, the costs, and watch the strategy recompute live.

🔗 Site + paper + code in the first comment.

Swipe through the carousel for the 5 charts that tell the story 👇

#Quant #QuantitativeFinance #DataScience #Python #Investing #MachineLearning #Finance

---

## 💬 First comment (paste as the first comment, both languages)

Links / Liens 👇
• Interactive site / Site interactif : YOUR-SITE-URL
• Paper (PDF) / Papier : YOUR-SITE-URL/paper/main.pdf
• Code (GitHub) : https://github.com/2Lucid/Trade-the-Risk-Not-the-Price

Built with Python (pandas, numpy, statsmodels, arch, hmmlearn). Research & education only — not investment advice.

---

## 🔁 Post 2 — the follow-up (mini-series), a few days later

### 🇬🇧 English
I gave an algorithm 26 years of market returns and **no crisis dates**. It found the crises by itself.

This is a Hidden Markov Model — it splits history into "calm" and "turbulent" regimes with zero supervision. The red bands below? Dot-com, 2008, COVID, 2022. Nobody told it when they happened.

Why it matters: it's not just a pretty chart. It explains *where* a volatility-managed strategy earns its keep — it cuts risk precisely in the red zones (vol ~21% → ~16%) and leans in during the green ones.

One subtlety most people miss: for a picture you can use the full-sample (smoothed) regimes, but the moment you trade on them you may only use the *filtered* (online) signal — otherwise you're cheating with hindsight. I implemented both.

Part 2 of my volatility-managed portfolios project. Site + paper + code in the comments.

#Quant #MachineLearning #Finance #DataScience #Python

### 🇫🇷 Français
J'ai donné à un algorithme 26 ans de rendements de marché et **aucune date de crise**. Il a trouvé les crises tout seul.

C'est un Hidden Markov Model : il découpe l'histoire en régimes « calme » et « agité » sans aucune supervision. Les bandes rouges ci-dessous ? Bulle internet, 2008, COVID, 2022. Personne ne lui a dit quand elles ont eu lieu.

Pourquoi c'est important : ce n'est pas qu'un joli graphique. Ça explique *où* une stratégie vol-managed gagne sa vie — elle coupe le risque précisément dans les zones rouges (vol ~21 % → ~16 %) et s'expose dans les vertes.

Une subtilité que beaucoup ratent : pour une figure on peut utiliser les régimes full-sample (lissés), mais dès qu'on trade dessus on ne peut utiliser que le signal *filtré* (online) — sinon on triche avec le futur. J'ai implémenté les deux.

Partie 2 de mon projet de portefeuilles vol-managed. Site + papier + code en commentaire.

#Quant #MachineLearning #Finance #DataScience #Python

---

## 🖼️ The carousel (already generated → `carousel.pdf`)

7 slides. Captions for reference:
1. **Cover** — "You can't predict the price. You can predict the risk."
2. **01 · The honest test** — return R² ≈ –12% vs volatility R² ≈ +28% (`fig01`)
3. **02 · The result** — managed vs buy-and-hold equity + drawdowns (`fig05`)
4. **03 · It explains itself** — the HMM regime chart (`fig07`)
5. **04 · Does it travel?** — cross-asset Sharpe, BH vs managed vs +trend (`fig09`)
6. **05 · The powerful part** — diversified managed+trend equity curve (`fig10`)
7. **CTA** — site + paper + GitHub

Individual slides are in `linkedin/slides/` if you'd rather post images than the PDF.
Regenerate after any change with: `python -m src.make_social`

---

## 📌 Posting tips

- **Format:** a *Document* post (the PDF carousel) gets strong reach and dwell time.
- **First line is everything** — LinkedIn cuts off after ~2 lines; the hook must land before "…see more".
- **Links in the first comment**, not the body (avoids the link penalty).
- **Best time:** Tue–Thu, ~8–10am your audience's time.
- **Reply to every comment in the first hour** — it compounds reach.
- Tag 1–2 relevant people/communities only if genuinely relevant (don't spam).
- The disclaimer ("research, not advice") also signals maturity to recruiters.
