#!/usr/bin/env python

from base.stringmatch import match


def test_match():
    m = match('hello', 'hello')
    assert m.cost == 0
    assert m.start_pos == 0
    assert m.end_pos == 5

    m = match('e', 'hello')
    assert m.cost == 0
    assert m.start_pos == 1
    assert m.end_pos == 2

    m = match('hello', 'e')
    assert m.cost == 4
    assert m.start_pos == 0
    assert m.end_pos == 1

    # Prefer character omissions over character edits in match bounds
    m = match('bab', 'cac')
    assert m.cost == 2
    assert m.start_pos == 1
    assert m.end_pos == 2

    # Select first match in the text in case of ties
    m = match('ab', 'ba')
    assert m.cost == 1
    assert m.start_pos == 0
    assert m.end_pos == 1

    m = match('hello', 'world')
    assert m.cost == 4
    assert m.start_pos == 1
    assert m.end_pos == 2


def test_unicode_match():
    m = match('æther', 'aether')
    assert m.cost == 1
    assert m.start_pos == 2
    assert m.end_pos == 6

    m = match('こんにちは世界', 'こんばんは世界')
    assert m.cost == 2
    assert m.start_pos == 0
    assert m.end_pos == 7


def test_long_match():
    caption = 'The crystal packing in cis-[Cr(phen)2F2]ClO4. H2O. Displacement ellipsoids are drawn at 50% probability. H atoms except the one originated from cystal water have been omitted.'
    page = ' supplementary materials Figures Fig. 1. The molecular structure and atom labeling scheme of cis-[Cr(phen) 2 F 2 ]ClO 4. H 2 O. Displacement ellipsoids are drawn at 50% probability. H atoms with arbitrary radii. Fig. 2. The crystal packing in cis-[Cr(phen) 2 F 2 ]ClO 4. H 2 O. Displacement ellipsoids are drawn at 50% probability. H atoms except the one originated from cystal water have been omitted. cis-Difluorido(1,10-phenanthroline)chromium(III) perchlorate monohydrate Crystal data [CrF 2 (C 12 H 8 N 2 ) 2 ]ClO 4 *H 2 O Z = 2 M r = 567.87 F 000 = 578 Triclinic, P1 D x = 1.651 Mg m -3 Hall symbol: -P 1 a = 7.6930 (10) A b = 9.4640 (8) A c = 16.0610 (17) A  = 79.750 (7)  = 83.228 (12)  = 88.115 (8) Mo K radiation  = 0.71073 A Cell parameters from 26598 reflections  = 2.3-25.0  = 0.68 mm -1 T = 122 (1) K Block, red 0.44 x 0.41 x 0.16 mm V = 1142.6 (2) A 3 Data collection Nonius KappaCCD area-detector diffractometer Radiation source: fine-focus sealed tube 4014 independent reflections Monochromator: graphite 3851 reflections with I  2(I) R int = 0.025 T = 122.0(10) K  max = 25.0  and  scans  min = 2.3 Absorption correction: gaussian integration (Coppens, 1970) T min = 0.794, T max = 0.913 28606 measured reflections sup-2 h = -99 k = -1111 l = -1819 '
    m = match(caption, page)
    assert m.cost == 7
    assert page[m.start_pos:m.end_pos] == 'The crystal packing in cis-[Cr(phen) 2 F 2 ]ClO 4. H 2 O. Displacement ellipsoids are drawn at 50% probability. H atoms except the one originated from cystal water have been omitted.'


def test_long_unicode_match():
    caption = "Factors influencing λ lysis time stochasticity. (A) Effect of allelic variation in holin proteins on mean lysis times (MLTs) and standard deviations (SDs). (B) Effect of λ's late promoter pR' activity [50] on MLTs, SDs and CVs (coefficients of variation). Solid curve is SD = 3.05 (72.73 + P)/P, where P was the pR' activity. (C) Effects of pR' activity and host growth rate on lysis time stochasticity. The regression line was obtained by fitting all data points from the late promoter activity (filled diamonds) and lysogen growth rate (open squares) treatments, except for the datum with the longest MLT and largest SD (from SYP028 in Table 2). (D) Effect of lysogen growth rate on MLT, SD, and CV. The fitted solid line shows the relationship between the growth rate and SD. All data are from Tables 1 and 2. Symbols: open circles, MLT; close circles, SD; closed triangles, CV."
    page = "Dennehy and Wang BMC Microbiology 2011, 11:174 http://www.biomedcentral.com/1471-2180/11/174 Page 4 of 12 Effect of allelic variation in holin sequence Table 1 Effects of holin allelic sequences on the stochasticity of lysis time a a Strain n MLT (min) IN61 274 45.7 2.92 IN56 (WT) 230 65.1 3.24 IN160 IN62 47 136 29.5 54.3 3.28 3.42 IN70 52 54.5 3.86 IN57 53 47.0 4.25 IN69 119 45.0 4.38 IN63 209 41.2 4.55 IN64 63 48.4 4.60 IN68 153 54.1 5.14 IN66 IN67 189 212 82.2 57.6 5.87 6.71 IN65 33 83.8 6.95 IN71 49 68.8 7.67 SD (min) In some cases, the sample size n is the pooled number of cells observed across several days. Detailed information can be found in Table S1 of additional file 1. It has long been known that different holin alleles show different lysis times [37,46,47]. However, it is not clear to what extent allelic differences in holin protein would affect the lysis timing of individual cells. To gain further insight, we determined the MLTs (mean lysis times) and SDs (standard deviations) of lysis time for 14 isogenic l lysogens differing in their S holin sequences (see APPENDIX B for our rationale for using SD as the mea- sure for lysis time stochasticity). The directly observed MLTs (Table 1) were longer than those reported pre- viously [46]. This discrepancy was mainly due to the fact that, in previous work, lysis time was defined by the time point when the turbidity of the lysogen culture began to decline, whereas in our current measurement, it was the mean of all individual lysis times observed for a particular phage strain. Figure 3A revealed a significant positive relationship between MLT and SD (F [1,12] = 8.42, p = 0.0133). How- ever, we did not observe a significant relationship Figure 3 Factors influencing l lysis time stochasticity. (A) Effect of allelic variation in holin proteins on mean lysis times (MLTs) and standard deviations (SDs). (B) Effect of l's late promoter p R ' activity [50] on MLTs, SDs and CVs (coefficients of variation). Solid curve is SD = 3.05 (72.73 + P)/P, where P was the p R ' activity. (C) Effects of p R ' activity and host growth rate on lysis time stochasticity. The regression line was obtained by fitting all data points from the late promoter activity (filled diamonds) and lysogen growth rate (open squares) treatments, except for the datum with the longest MLT and largest SD (from SYP028 in Table 2). (D) Effect of lysogen growth rate on MLT, SD, and CV. The fitted solid line shows the relationship between the growth rate and SD. All data are from Tables 1 and 2. Symbols: open circles, MLT; close circles, SD; closed triangles, CV."
    m = match(caption, page)
    assert m.cost == 8
    assert page[m.start_pos:m.end_pos] == "Factors influencing l lysis time stochasticity. (A) Effect of allelic variation in holin proteins on mean lysis times (MLTs) and standard deviations (SDs). (B) Effect of l's late promoter p R ' activity [50] on MLTs, SDs and CVs (coefficients of variation). Solid curve is SD = 3.05 (72.73 + P)/P, where P was the p R ' activity. (C) Effects of p R ' activity and host growth rate on lysis time stochasticity. The regression line was obtained by fitting all data points from the late promoter activity (filled diamonds) and lysogen growth rate (open squares) treatments, except for the datum with the longest MLT and largest SD (from SYP028 in Table 2). (D) Effect of lysogen growth rate on MLT, SD, and CV. The fitted solid line shows the relationship between the growth rate and SD. All data are from Tables 1 and 2. Symbols: open circles, MLT; close circles, SD; closed triangles, CV."


if __name__ == '__main__':
    import pytest

    pytest.main([__file__])
