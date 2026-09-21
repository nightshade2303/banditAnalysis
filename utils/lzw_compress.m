function [result, ratio, alphabet, randomRatio, randomLengths] = lzw_compress(uncompressed, varargin)
%LZW_COMPRESS Compress a string using a simple LZW-style dictionary.
%
%   result = lzw_compress(uncompressed)
%   [result, ratio] = lzw_compress(uncompressed, 'ReturnRatio', true)
%   [result, ratio, alphabet, randomRatio, randomLengths] = ...
%       lzw_compress(uncompressed, 'ReturnRatio', true, ...
%                               'ReturnAlphabet', true, ...
%                               'ReturnRandomRatio', true, ...
%                               'ReturnRandomLengths', true)
%
%   The random-baseline path samples strings of the same length as the
%   input using the fixed symbols '1', '2', '3', and '4'.

    p = inputParser;
    p.FunctionName = 'lzw_compress';
    addRequired(p, 'uncompressed');
    addParameter(p, 'ReturnRatio', false);
    addParameter(p, 'ReturnAlphabet', false);
    addParameter(p, 'ReturnRandomRatio', false);
    addParameter(p, 'ReturnRandomLengths', false);
    addParameter(p, 'NRandomSamples', 100);
    addParameter(p, 'Rng', []);
    parse(p, uncompressed, varargin{:});

    opts = p.Results;

    if isstring(uncompressed)
        uncompressed = char(uncompressed);
    end
    uncompressed = char(uncompressed);

    [result, alphabet] = compressCodes(uncompressed);

    ratio = [];
    if opts.ReturnRatio
        if isempty(uncompressed)
            ratio = 1.0;
        else
            ratio = numel(uncompressed) / numel(result);
        end
    end

    randomRatio = [];
    randomLengths = [];
    if opts.ReturnRandomRatio || opts.ReturnRandomLengths
        if opts.NRandomSamples < 1
            error('NRandomSamples must be at least 1.');
        end

        if isempty(opts.Rng)
            rngState = rng;
        else
            rngState = rng(opts.Rng);
        end
        %#ok<NASGU>

        baselineSymbols = '1234';
        inputLen = numel(uncompressed);
        originalLen = numel(result);
        randomLengths = zeros(opts.NRandomSamples, 1);

        for sampleIdx = 1:opts.NRandomSamples
            if inputLen == 0
                randomSeq = '';
            else
                randomIdx = randi(numel(baselineSymbols), [1, inputLen]);
                randomSeq = baselineSymbols(randomIdx);
            end

            randomCodes = compressCodes(randomSeq);
            randomLengths(sampleIdx) = numel(randomCodes);
        end

        if opts.ReturnRandomRatio
            if originalLen == 0
                randomRatio = 1.0;
            else
                randomRatio = mean(randomLengths / originalLen);
            end
        end

        if ~opts.ReturnRandomLengths
            randomLengths = [];
        end
    end

    if ~opts.ReturnAlphabet
        alphabet = [];
    end
end

function [codes, alphabet] = compressCodes(uncompressed)
    if isempty(uncompressed)
        codes = [];
        alphabet = '';
        return;
    end

    alphabet = unique(uncompressed, 'stable');
    dictionary = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for idx = 1:numel(alphabet)
        dictionary(alphabet(idx)) = idx - 1;
    end

    dictSize = numel(alphabet);
    codes = [];
    w = '';

    for idx = 1:numel(uncompressed)
        c = uncompressed(idx);
        wc = [w c];
        if isKey(dictionary, wc)
            w = wc;
        else
            codes(end + 1) = dictionary(w); %#ok<AGROW>
            dictionary(wc) = dictSize;
            dictSize = dictSize + 1;
            w = c;
        end
    end

    if ~isempty(w)
        codes(end + 1) = dictionary(w); %#ok<AGROW>
    end
end
