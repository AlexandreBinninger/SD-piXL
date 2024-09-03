import React, { useState, useEffect } from 'react';
import { Box, Text, Button, useClipboard } from '@chakra-ui/react';

export default function Bibtex({path}) {
  const [bibContent, setBibContent] = useState('');
  const { hasCopied, onCopy } = useClipboard(bibContent);

  useEffect(() => {
    const fetchBib = async () => {
      const response = await fetch(path);
      const text = await response.text();
      setBibContent(text);
    };

    fetchBib();
  }, [path]);

  return (
    <Box p={4} borderWidth="1px" borderRadius="lg">
      <Text fontFamily="monospace" whiteSpace="pre-wrap">{bibContent}</Text>
      <Button mt={4} onClick={onCopy} size="sm">
        {hasCopied ? 'Copied' : 'Copy Citation'}
      </Button>
    </Box>
  );
};
