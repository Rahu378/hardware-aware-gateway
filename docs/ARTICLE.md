# The Fastest Code I Wrote Last Week Was the Code I Didn't Write

### I made an AI model generate text 1.7x faster. The optimization everyone told me to build would have gained less than 6%, and I only found that out because I did the arithmetic first.

---

There is a kitchen I want you to picture, because it is the whole story.

A chef stands at the pass. The chef is extraordinary: chops, sears, and plates faster than you can follow. Next to the chef is a waiter whose only job is handing over instruction slips. *Chop the onion. Now the garlic. Now stir.*

One dish takes six thousand slips.

The waiter needs about three milliseconds to deliver each one. The chef needs roughly fifteen seconds of actual cooking. The dish takes thirty-three seconds to leave the kitchen.

**The chef spends more than half the service standing still, holding a knife, waiting for a piece of paper.**

Everyone who walks into that kitchen says the same thing. *You need a faster chef.* Sharper knives. A hotter stove. A better pan.

I measured the chef. The chef was already moving at 87% of what physics allows.

The fix was to hand the chef the entire recipe on one page, once, and never speak again.

---

## What this actually was

The chef is a GPU. The waiter is a CPU. The dish is one word of text coming out of an AI model.

I spent a week on a project called [Hardware-Aware Gateway](https://github.com/Rahu378/hardware-aware-gateway): take an open language model, find out precisely where its time goes, write custom GPU code to fix it, and publish the before-and-after honestly.

The plan I started from was the standard one. Profile the model. Find the memory bottleneck. Write a fast custom kernel for one layer. Document the speedup.

I did all of that. Two of the three most useful results were decisions **not** to write code.

---

## Act One: I did the obvious thing, and it made everything slower

Language models spend enormous effort moving numbers around rather than doing arithmetic on them. A useful trick is *fusion*: instead of five small operations that each read from memory and write back, you write one operation that does all five in a single pass. Fewer trips, less waiting.

I wrote two fused operations. Twice each, in fact — once in Triton for NVIDIA GPUs, once as hand-written Metal kernels for Apple silicon, so I could compare two very different machines.

In isolation, they were fast. One hit **5.7x faster than the standard implementation**, sustaining 229 GB/s of memory bandwidth, which is 95% of everything that GPU's memory system can physically deliver. That is close enough to the wall that there is nothing meaningful left.

Then I plugged them into a real model and measured the whole thing end to end.

**It got slower.** 0.83x. A 17% regression.

![Decode time split into GPU work and CPU waiting](https://raw.githubusercontent.com/Rahu378/hardware-aware-gateway/main/docs/img/where-the-time-goes.png)

The profiler explained it in about thirty seconds. Matrix multiplication was **76% of the GPU's time**. Every operation my fused kernels touched added up to **19%**. So even a flawless fusion had a ceiling of 19%, and mine had a hidden cost that ate more than it saved.

Here is the specific embarrassment. When the model generates text one word at a time, each step processes a *single row* of numbers. My fused kernel was 5.7x faster on two thousand rows and **0.42x on one row** — less than half the speed of the code it replaced. Launching a custom kernel has a fixed cost, and at one row that cost was larger than everything the fusion saved.

Fusion is a bandwidth optimization. At one row, nothing is bandwidth-limited. I had brought the right tool to the wrong problem.

![Speedup against row count, showing where fusion drops below break-even](https://raw.githubusercontent.com/Rahu378/hardware-aware-gateway/main/docs/img/fusion-crossover.png)

The fix was small: below a threshold, don't use the fused kernel. But I refused to guess the threshold. I measured it — fusion loses at 8 rows, wins at 32 — and the code now measures it again on whatever machine it lands on, because that number is a property of the hardware, not of the kernel.

> A 5.7x speedup on 19% of the work is a 5.7x speedup on 19% of the work. The number that decides anything is the one at the end.

---

## Act Two: the kernel I didn't write

With the regression fixed, the next target looked obvious. Matrix multiplication was 76% of GPU time. Everyone optimizes matrix multiplication. There are famous blog posts about it.

Before writing a line, I did the arithmetic.

When a model generates one word, it must read **every single weight** in the model out of memory. For the model I was using that is 3.09 GB, every word. That GPU's memory delivers 241 GB/s, measured, not from a datasheet.

3.09 GB divided by 241 GB/s is **12.8 milliseconds**. That is the floor. No code of any kind beats it, because the bytes have to physically arrive.

I measured what NVIDIA's own library was already achieving: **14.6 milliseconds**. It was running at 87% of the physical wall.

So a perfect, flawless, world-class hand-written replacement — better than a library NVIDIA has tuned for a decade — would have recovered **1.8 milliseconds out of a 32.0 millisecond word. Under 6%.**

I did not write it. That decision is in the repository, with the numbers that justify it.

> The most expensive thing in engineering is not slow code. It is a week spent making something faster that was never the reason things were slow.

---

## Act Three: where the time was actually going

If the arithmetic takes 14.6 ms and the word takes 32.0 ms, something is consuming **17.4 milliseconds** doing nothing visible.

The GPU was idle. Waiting.

Every operation a model performs has to be *dispatched* — the CPU tells the GPU what to do next. Each instruction is cheap, about 2.84 microseconds. But generating one word takes **6,130 of them**.

6,130 x 2.84 microseconds = **17.4 milliseconds**, spent entirely on a CPU issuing paperwork while an idle GPU waits.

That is the waiter. That is 54% of every word.

**No kernel fixes this.** You can make the GPU infinitely fast and the CPU still needs 17.4 ms to describe the work.

The fix is a CUDA graph. Instead of issuing six thousand instructions per word forever, you record the sequence once and replay the recording. One submission instead of six thousand.

**28.0 words per second became 48.1. A 1.72x speedup.** More than every fused kernel in the project combined.

---

## The part I am actually proud of

Before writing any of the graph code, the analysis said the dispatch gap was **17.4 milliseconds per word**.

After it worked, the graph had recovered **14.9 milliseconds**.

**86% of a number predicted in advance, in real units, before the code existed.**

That is the difference between optimizing and guessing. The same arithmetic that predicted this win is what told me not to write the matrix multiplication kernel. One analysis, one afternoon, and it both saved a week and pointed at the thing that actually worked.

There is a detail I want to be precise about, because it is the sort of thing that quietly ruins projects like this. A CUDA graph that replays against the wrong memory does not crash. It produces text that is completely fluent and completely wrong, and nothing anywhere raises an error. So the benchmark refuses to report a speed number until the graphed version has produced **token-for-token identical output** to the original. Greedy text generation is deterministic; identical is the only acceptable answer, not "close."

---

## Two more things I measured and then didn't build

**Recording the prompt-processing stage too.** Sounds obviously worth doing. It is worth **1.006x**, and the reason is elegant: the dispatch cost is *fixed* (6,130 instructions regardless of prompt length) while the actual work grows with the prompt. On a short prompt the overhead is 40% of that stage but the stage is 3% of the request. On a long prompt the stage dominates the request but the overhead has shrunk to 1% of it. Squeezed from both ends, at every size from 128 to 8,192 words.

**FlashAttention**, the famous optimization everyone reaches for. In this workload, attention was **1.1% of GPU time**. Perfect attention would have been invisible.

Neither of those is a criticism of the technique. They are statements about *this* workload — which is the only kind of statement a measurement can make.

---

## What surprised me most: the ruler was broken

For three separate runs, I could not prove the fused kernels helped end to end. The effect was around 5%, and my statistics kept saying "not resolved, take more samples."

So I took more samples. Sixteen. Still nothing. It suggested twenty-one.

Then I ran the identical benchmark on a different free platform. Same code, same model, same GPU model.

The measurement noise on the first platform was **3.68**. On the second it was **0.53** — seven times quieter. The effect resolved immediately, with overwhelming confidence.

The first platform's shared virtual machine was bimodal: identical code clustered around two different speeds depending on what else was running on that host. **You cannot average your way out of a broken instrument.** More samples of a wobbling ruler give you a very precise measurement of the wobble.

> When a test keeps failing and keeps asking for more data, suspect the instrument before the data.

---

## Try it yourself

Everything is public, and every number in the repository is generated from committed measurement files. There is a build check that fails if the README ever disagrees with the raw data, because I did not trust myself to keep them in sync by hand — and it caught me twice.

**[Interactive explorer](https://rahu378.github.io/hardware-aware-gateway/)** — pick a GPU, a model size, and a precision, and it tells you whether you are limited by memory or by the CPU, and what to do about each. No signup, no GPU needed. It reproduces my measured results to within 7%.

**[The repository](https://github.com/Rahu378/hardware-aware-gateway)** — kernels, benchmarks, profiling, and the full write-up.

**[Run the whole thing on a free GPU](https://colab.research.google.com/github/Rahu378/hardware-aware-gateway/blob/main/notebooks/colab_bootstrap.ipynb)** — about ten minutes, and it regenerates every figure in this article on your own hardware.

There is also a side-by-side demo where the original and the optimized version generate the same text at the same time, with live speed counters. They produce identical words; one just finishes first.

---

## The thing worth taking away

I could have shipped the 5.7x number. It is true, it is measured, and it would have looked excellent on a slide.

It also would have been useless, because the system it lived in got slower.

The three findings I would actually defend in a room full of engineers are these: **the optimization I built made things worse and I published that**, **the optimization I was told to build next was worth under 6% and I calculated that before spending the week**, and **the thing that actually worked was invisible until I stopped looking at the GPU and started looking at what was keeping it idle**.

Fast code is not the hard part. Knowing which code is worth making fast is the hard part.

---

*The GPU work here is in Triton and Metal, on a Tesla T4 and an Apple M3, using Qwen2.5-1.5B. All of it runs on free hardware. Code and data: [github.com/Rahu378/hardware-aware-gateway](https://github.com/Rahu378/hardware-aware-gateway)*
