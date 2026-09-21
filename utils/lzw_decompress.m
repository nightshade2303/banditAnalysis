function text = lzw_decompress(compressed, alphabet)
%LZW_DECOMPRESS Decompress integer codes produced by lzw_compress.
%
%   text = lzw_decompress(compressed)
%   text = lzw_decompress(compressed, alphabet)
%
%   If alphabet is omitted, a legacy 256-entry ASCII dictionary is used.

    if nargin < 2 || isempty(alphabet)
        dictSize = 256;
        dictionary = cell(1, dictSize);
        for idx = 1:dictSize
            dictionary{idx} = char(idx - 1);
        end
    else
        if isstring(alphabet)
            alphabet = char(alphabet);
        end
        if iscell(alphabet)
            alphabet = [alphabet{:}];
        end

        dictSize = numel(alphabet);
        dictionary = cell(1, dictSize);
        for idx = 1:dictSize
            dictionary{idx} = alphabet(idx);
        end
    end

    compressed = compressed(:).';
    if isempty(compressed)
        text = '';
        return;
    end

    firstCode = compressed(1);
    if firstCode < 0 || firstCode >= numel(dictionary) || isempty(dictionary{firstCode + 1})
        error('Bad first compressed code: %d', firstCode);
    end

    w = dictionary{firstCode + 1};
    parts = {w};

    for idx = 2:numel(compressed)
        code = compressed(idx);
        if code >= 0 && code < numel(dictionary) && ~isempty(dictionary{code + 1})
            entry = dictionary{code + 1};
        elseif code == dictSize
            entry = [w w(1)];
        else
            error('Bad compressed code: %d', code);
        end

        parts{end + 1} = entry; %#ok<AGROW>
        dictionary{dictSize + 1} = [w entry(1)];
        dictSize = dictSize + 1;
        w = entry;
    end

    text = [parts{:}];
end
