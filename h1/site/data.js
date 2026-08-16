window.H1_DATA = {
  s04: [
    {name:"Fresh", acc:35.7, tok:109},
    {name:"Full transcript", acc:81.0, tok:499},
    {name:"Deterministic summary", acc:61.9, tok:274},
    {name:"Model summary", acc:69.0, tok:469},
    {name:"Structured state", acc:64.3, tok:371},
    {name:"Combined", acc:66.7, tok:730}
  ],
  s05: {
    deterministic:{retention:100, terminal:100, goal:100},
    delta:{retention:13.2, terminal:11.1, goal:42.8},
    full:{retention:6.3, terminal:0, goal:16.7}
  },
  s06: {
    pooled:{incremental:60.4,replay:59.4,transcript:67.7,reconstructed:39.6,fresh:27.1},
    horizons:[
      {t:10, incremental:65.6, transcript:71.9, incTok:418.4, transTok:576.2},
      {t:25, incremental:56.2, transcript:71.9, incTok:419.5, transTok:782.4},
      {t:50, incremental:59.4, transcript:59.4, incTok:424.7, transTok:1063.6}
    ]
  }
};